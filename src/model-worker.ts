/**
 * Web Worker wrapper for segmentation model inference.
 *
 * Moves MediaPipe inference off the main thread so the ~13ms model run
 * never blocks UI or GPU rendering. Communication via transferable objects
 * (ImageBitmap in, Float32Array buffer out) for zero-copy overhead.
 *
 * Usage:
 *   const worker = new ModelWorkerClient(config);
 *   await worker.init();
 *   worker.requestSegment(imageBitmap, timestamp, crop);
 *   // mask arrives asynchronously via onMask callback
 */

import type { ModelConfig, CropRegion } from './model';

// Worker code as string — loaded via Blob URL so no separate file is needed.
// Uses dynamic import() for MediaPipe to work in both classic and module workers.
const WORKER_SOURCE = `
'use strict';

let segmenter = null;
let resizeCanvas = null;
let resizeCtx = null;
let config = null;
let lastMask = null;
let previousFullMask = null;
let hasPreviousMask = false;

async function init(cfg) {
  config = cfg;

  const mp = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/+esm'
  );

  const vision = await mp.FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'
  );

  const opts = {
    baseOptions: {
      modelAssetPath: cfg.modelAssetPath,
      delegate: cfg.delegate,
    },
    runningMode: 'VIDEO',
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  };

  let actualDelegate = cfg.delegate;
  try {
    segmenter = await mp.ImageSegmenter.createFromOptions(vision, opts);
  } catch (e) {
    if (cfg.delegate === 'GPU') {
      console.warn('[segmo worker] GPU delegate failed, falling back to CPU:', e);
      opts.baseOptions.delegate = 'CPU';
      segmenter = await mp.ImageSegmenter.createFromOptions(vision, opts);
      actualDelegate = 'CPU';
    } else {
      throw e;
    }
  }

  resizeCanvas = new OffscreenCanvas(cfg.outputWidth, cfg.outputHeight);
  resizeCtx = resizeCanvas.getContext('2d', { willReadFrequently: true });

  self.postMessage({ type: 'ready', actualDelegate });
}

function segment(bitmap, timestamp, crop) {
  if (!segmenter || !resizeCtx) {
    bitmap.close();
    return;
  }

  const ow = config.outputWidth;
  const oh = config.outputHeight;

  if (crop) {
    const bw = bitmap.width;
    const bh = bitmap.height;
    resizeCtx.drawImage(bitmap,
      crop.x * bw, crop.y * bh, crop.w * bw, crop.h * bh,
      0, 0, ow, oh);
  } else {
    resizeCtx.drawImage(bitmap, 0, 0, ow, oh);
  }
  bitmap.close();

  // Pass ImageData instead of OffscreenCanvas — MediaPipe's segmentForVideo
  // silently returns empty masks when given OffscreenCanvas in a worker context.
  // ImageData is universally supported and just raw pixels.
  const imageData = resizeCtx.getImageData(0, 0, ow, oh);

  const modelStart = performance.now();
  let result;
  try {
    result = segmenter.segmentForVideo(imageData, timestamp);
  } catch (e) {
    //console.error('[segmo worker] segmentForVideo failed:', e);
    return;
  }

  if (!result.confidenceMasks || result.confidenceMasks.length === 0) {
    //console.warn('[segmo worker] segmentForVideo returned no confidence masks');
    return;
  }

  const firstMask = result.confidenceMasks[0];
  // getAsFloat32Array() forces GPU readback — must be inside timing window
  const firstData = firstMask.getAsFloat32Array();
  const pixelCount = firstData.length;

  // Reuse buffers
  if (!lastMask || lastMask.length !== pixelCount) {
    lastMask = new Float32Array(pixelCount);
    previousFullMask = new Float32Array(pixelCount);
  }

  if (result.confidenceMasks.length > 2) {
    for (let i = 0; i < pixelCount; i++) lastMask[i] = 1.0 - firstData[i];
  } else {
    const mask = result.confidenceMasks[result.confidenceMasks.length - 1];
    const maskData = mask.getAsFloat32Array();
    lastMask.set(maskData);
    if (mask !== firstMask) mask.close();
  }
  const inferenceMs = performance.now() - modelStart;

  firstMask.close();
  for (let i = 1; i < result.confidenceMasks.length; i++) {
    result.confidenceMasks[i].close();
  }

  // Map crop mask to full frame + compute bbox
  let outMask = lastMask;
  let bMinX = ow, bMinY = oh, bMaxX = 0, bMaxY = 0, bFound = false;

  if (crop) {
    outMask = new Float32Array(ow * oh);
    const cx0 = crop.x * ow, cy0 = crop.y * oh;
    const cw = crop.w * ow, ch = crop.h * oh;
    const x0 = Math.max(0, Math.floor(cx0));
    const y0 = Math.max(0, Math.floor(cy0));
    const x1 = Math.min(ow, Math.ceil(cx0 + cw));
    const y1 = Math.min(oh, Math.ceil(cy0 + ch));
    const scaleX = ow / cw, scaleY = oh / ch;

    for (let y = y0; y < y1; y++) {
      const sy = Math.min(((y - cy0) * scaleY) | 0, oh - 1);
      const yOff = y * ow, syOff = sy * ow;
      for (let x = x0; x < x1; x++) {
        const sx = Math.min(((x - cx0) * scaleX) | 0, ow - 1);
        const val = lastMask[syOff + sx];
        outMask[yOff + x] = val;
        if (val > 0.5) {
          if (x < bMinX) bMinX = x;
          if (x > bMaxX) bMaxX = x;
          if (y < bMinY) bMinY = y;
          if (y > bMaxY) bMaxY = y;
          bFound = true;
        }
      }
    }
  } else {
    outMask = new Float32Array(lastMask);
    for (let y = 0; y < oh; y++) {
      const yOff = y * ow;
      for (let x = 0; x < ow; x++) {
        if (outMask[yOff + x] > 0.5) {
          if (x < bMinX) bMinX = x;
          if (x > bMaxX) bMaxX = x;
          if (y < bMinY) bMinY = y;
          if (y > bMaxY) bMaxY = y;
          bFound = true;
        }
      }
    }
  }

  // Compute motion map in full-frame space (after ROI mapping)
  const motion = new Float32Array(pixelCount);
  if (hasPreviousMask && previousFullMask) {
    for (let i = 0; i < pixelCount; i++) {
      motion[i] = Math.abs(outMask[i] - previousFullMask[i]);
    }
  }

  // Save previous mask in full-frame space for next frame's motion map
  if (!previousFullMask || previousFullMask.length !== pixelCount) {
    previousFullMask = new Float32Array(outMask);
  } else {
    previousFullMask.set(outMask);
  }
  hasPreviousMask = true;

  // Transfer buffers (zero-copy)
  self.postMessage({
    type: 'mask',
    mask: outMask.buffer,
    motion: motion.buffer,
    bbox: bFound ? { minX: bMinX, minY: bMinY, maxX: bMaxX, maxY: bMaxY } : null,
    inferenceMs,
  }, [outMask.buffer, motion.buffer]);
}

self.onmessage = (e) => {
  if (e.data.type === 'init') init(e.data.config);
  if (e.data.type === 'segment') segment(e.data.bitmap, e.data.timestamp, e.data.crop);
};
`;

export interface WorkerMaskResult {
  mask: Float32Array;
  motion: Float32Array;
  bbox: { minX: number; minY: number; maxX: number; maxY: number } | null;
  inferenceMs: number;
}

export class ModelWorkerClient {
  private worker: Worker | null = null;
  private ready = false;
  private config: Required<ModelConfig>;
  private onMask: ((result: WorkerMaskResult) => void) | null = null;
  private blobUrl: string | null = null;
  /** The delegate actually used (GPU may fall back to CPU) */
  actualDelegate: 'GPU' | 'CPU' = 'CPU';

  constructor(config: ModelConfig) {
    this.config = {
      modelAssetPath: config.modelAssetPath || 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite' ||
        'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
      delegate: config.delegate || 'GPU',
      outputWidth: config.outputWidth || 256,
      outputHeight: config.outputHeight || 144,
    };
  }

  /** Register callback for when worker produces a new mask */
  onMaskReady(cb: (result: WorkerMaskResult) => void): void {
    this.onMask = cb;
  }

  /** Initialize the worker and load MediaPipe model */
  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const blob = new Blob([WORKER_SOURCE], { type: 'text/javascript' });
        this.blobUrl = URL.createObjectURL(blob);
        this.worker = new Worker(this.blobUrl);

        const timeout = setTimeout(() => {
          reject(new Error('Worker init timed out (30s)'));
        }, 30000);

        this.worker.onmessage = (e) => {
          if (e.data.type === 'ready') {
            this.ready = true;
            this.actualDelegate = e.data.actualDelegate ?? this.config.delegate;
            clearTimeout(timeout);
            resolve();
          } else if (e.data.type === 'mask') {
            this.onMask?.({
              mask: new Float32Array(e.data.mask),
              motion: new Float32Array(e.data.motion),
              bbox: e.data.bbox,
              inferenceMs: e.data.inferenceMs ?? 0,
            });
          }
        };

        this.worker.onerror = (e) => {
          clearTimeout(timeout);
          reject(new Error(`Worker error: ${e.message}`));
        };

        this.worker.postMessage({
          type: 'init',
          config: this.config,
        });
      } catch (e) {
        reject(e);
      }
    });
  }

  /** Send a frame to the worker for inference (non-blocking) */
  requestSegment(
    frame: ImageBitmapSource,
    timestamp: number,
    crop?: CropRegion | null,
  ): void {
    if (!this.worker || !this.ready) return;

    // createImageBitmap is async but fast — just schedules the copy
    createImageBitmap(frame as ImageBitmapSource).then(bitmap => {
      this.worker?.postMessage(
        { type: 'segment', bitmap, timestamp, crop: crop ?? null },
        [bitmap], // transfer ownership (zero-copy)
      );
    }).catch(() => {
      // Frame capture failed (e.g., video not playing) — skip silently
    });
  }

  get isReady(): boolean {
    return this.ready;
  }

  destroy(): void {
    this.worker?.terminate();
    this.worker = null;
    this.ready = false;
    if (this.blobUrl) {
      URL.revokeObjectURL(this.blobUrl);
      this.blobUrl = null;
    }
  }
}
