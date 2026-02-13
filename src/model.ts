/**
 * Segmentation Model Wrapper
 *
 * Wraps MediaPipe's Image Segmenter for efficient inference.
 *
 * KEY DESIGN DECISION: Google Meet uses CPU inference (WASM + SIMD + XNNPACK),
 * NOT GPU. This is counterintuitive but intentional:
 * - CPU via WASM+SIMD has more predictable performance across devices
 * - GPU delegate can cause contention with WebGL rendering pipeline
 * - XNNPACK kernels are heavily optimized for these small models
 * - Frees the GPU entirely for the post-processing pipeline
 *
 * We default to CPU but allow GPU override for devices where it's faster.
 */

import {
  ImageSegmenter,
  FilesetResolver,
  ImageSegmenterResult,
} from '@mediapipe/tasks-vision';

export interface ModelConfig {
  /** Model asset path (default: MediaPipe selfie segmenter landscape) */
  modelAssetPath?: string;
  /**
   * Delegate: 'CPU' or 'GPU'
   * Default: 'CPU' — matches Google Meet's approach.
   * CPU with WASM+SIMD+XNNPACK is faster for these small models on most devices
   * and avoids GPU contention with the WebGL post-processing pipeline.
   */
  delegate?: 'GPU' | 'CPU';
  /** Output mask width (default: 256) */
  outputWidth?: number;
  /** Output mask height (default: 144) */
  outputHeight?: number;
}

const DEFAULT_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';

// Landscape model is optimized for 256x144 (16:9 video) — faster than the square model
const LANDSCAPE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite';

// Google Meet's HD segmentation model — higher quality boundaries
const MEET_HD_MODEL_URL = '/demo/segm_gpu_hd.tflite';

/** Normalized crop region (0-1 fractions of source frame) */
export interface CropRegion {
  x: number;
  y: number;
  w: number;
  h: number;
}

export class SegmentationModel {
  private segmenter: ImageSegmenter | null = null;
  private config: Required<ModelConfig>;
  private resizeCanvas: HTMLCanvasElement | OffscreenCanvas;
  private resizeCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  private lastMask: Float32Array | null = null;
  private previousMask: Float32Array | null = null; // For motion detection
  private fullMask: Float32Array | null = null; // Full-frame mask (ROI placed back)
  private lastCropRegion: CropRegion | null = null; // Crop used for last inference

  constructor(config: ModelConfig = {}) {
    this.config = {
      modelAssetPath: config.modelAssetPath || DEFAULT_MODEL_URL,
      delegate: config.delegate || 'CPU',
      outputWidth: config.outputWidth || 256,
      outputHeight: config.outputHeight || 256,
    };

    // Use HTMLCanvasElement in main thread — MediaPipe's segmentForVideo
    // doesn't accept OffscreenCanvas. Fall back to OffscreenCanvas in workers.
    if (typeof document !== 'undefined') {
      const canvas = document.createElement('canvas');
      canvas.width = this.config.outputWidth;
      canvas.height = this.config.outputHeight;
      this.resizeCanvas = canvas;
    } else {
      this.resizeCanvas = new OffscreenCanvas(this.config.outputWidth, this.config.outputHeight);
    }
    this.resizeCtx = this.resizeCanvas.getContext('2d', {
      willReadFrequently: false,
    })! as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  }

  /** Initialize the segmentation model */
  async init(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm',
    );

    this.segmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: this.config.modelAssetPath,
        delegate: this.config.delegate,
      },
      runningMode: 'VIDEO',
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
  }

  /**
   * Run segmentation on a video frame.
   *
   * @param frame - Camera frame to segment
   * @param timestamp - Frame timestamp in ms (performance.now())
   * @param crop - Optional ROI crop region (normalized 0-1). When provided,
   *               only the crop region is sent to the model at full 256x256 resolution,
   *               giving ~2x more detail on person boundaries.
   * @param frameWidth - Full frame width (needed for ROI mapping)
   * @param frameHeight - Full frame height (needed for ROI mapping)
   * @returns Float32Array confidence mask at model resolution (full-frame coordinates)
   */
  segment(
    frame: TexImageSource,
    timestamp: number,
    crop?: CropRegion | null,
    frameWidth?: number,
    frameHeight?: number,
  ): Float32Array | null {
    if (!this.segmenter) return this.fullMask ?? this.lastMask;

    const { outputWidth, outputHeight } = this.config;

    if (crop && frameWidth && frameHeight) {
      // ROI mode: draw only the crop region to fill the model canvas
      // This gives the model ~2x more resolution on the person's edges
      const sx = crop.x * frameWidth;
      const sy = crop.y * frameHeight;
      const sw = crop.w * frameWidth;
      const sh = crop.h * frameHeight;
      this.resizeCtx.drawImage(
        frame as CanvasImageSource,
        sx, sy, sw, sh,
        0, 0, outputWidth, outputHeight,
      );
      this.lastCropRegion = crop;
    } else {
      // Full-frame mode: resize entire frame to model input
      this.resizeCtx.drawImage(
        frame as CanvasImageSource,
        0, 0, outputWidth, outputHeight,
      );
      this.lastCropRegion = null;
    }

    // Run inference
    let result: ImageSegmenterResult;
    try {
      result = this.segmenter.segmentForVideo(
        this.resizeCanvas as HTMLCanvasElement,
        timestamp,
      );
    } catch (e) {
      console.error('[segmo] segmentForVideo failed:', e);
      return this.lastMask;
    }

    // Extract confidence mask (person vs background)
    // Multiclass model returns [background, hair, body-skin, face-skin, clothes, accessories]
    // For single-class model returns [background] or [person]
    if (result.confidenceMasks && result.confidenceMasks.length > 0) {
      const firstMask = result.confidenceMasks[0];
      const firstData = firstMask.getAsFloat32Array();
      const pixelCount = firstData.length;

      // Reuse buffers to minimize GC pressure
      if (!this.lastMask || this.lastMask.length !== pixelCount) {
        this.lastMask = new Float32Array(pixelCount);
        this.previousMask = new Float32Array(pixelCount);
      } else {
        this.previousMask!.set(this.lastMask);
      }

      if (result.confidenceMasks.length > 2) {
        // Multiclass: person = 1 - background (sum all non-background categories)
        // Background is index 0, so person confidence = 1 - background confidence
        for (let i = 0; i < pixelCount; i++) {
          this.lastMask[i] = 1.0 - firstData[i];
        }
      } else {
        // Single-class or binary: use last mask (person confidence)
        const mask = result.confidenceMasks[result.confidenceMasks.length - 1];
        const maskData = mask.getAsFloat32Array();
        this.lastMask.set(maskData);
        if (mask !== firstMask) mask.close();
      }

      // Close masks to release internal buffers
      firstMask.close();
      for (let i = 1; i < result.confidenceMasks.length; i++) {
        result.confidenceMasks[i].close();
      }
    }

    // Map crop-space mask back to full-frame coordinates
    if (this.lastCropRegion && this.lastMask) {
      const mw = outputWidth;
      const mh = outputHeight;
      const totalPixels = mw * mh;

      if (!this.fullMask || this.fullMask.length !== totalPixels) {
        this.fullMask = new Float32Array(totalPixels);
      }
      this.fullMask.fill(0); // background outside crop

      const crop = this.lastCropRegion;
      // Map each full-frame mask pixel to the crop-space mask
      // Crop region in mask coordinates
      const cx0 = crop.x * mw;
      const cy0 = crop.y * mh;
      const cw = crop.w * mw;
      const ch = crop.h * mh;

      for (let y = 0; y < mh; y++) {
        for (let x = 0; x < mw; x++) {
          // Is this pixel inside the crop region?
          const nx = (x - cx0) / cw; // normalized position within crop (0-1)
          const ny = (y - cy0) / ch;
          if (nx >= 0 && nx < 1 && ny >= 0 && ny < 1) {
            // Nearest-neighbor lookup in crop mask
            const sx = Math.min(Math.floor(nx * mw), mw - 1);
            const sy = Math.min(Math.floor(ny * mh), mh - 1);
            this.fullMask[y * mw + x] = this.lastMask[sy * mw + sx];
          }
        }
      }

      return this.fullMask;
    }

    return this.lastMask;
  }

  /** Get the last computed mask (for interpolated frames) */
  getLastMask(): Float32Array | null {
    return this.fullMask ?? this.lastMask;
  }

  /**
   * Get per-pixel motion estimate between last two masks.
   * Uses the full-frame mask (after ROI mapping) for consistency.
   */
  getMotionMap(): Float32Array | null {
    const current = this.fullMask ?? this.lastMask;
    if (!current || !this.previousMask) return null;

    const motion = new Float32Array(current.length);
    for (let i = 0; i < current.length; i++) {
      motion[i] = Math.abs(current[i] - this.previousMask[i]);
    }
    return motion;
  }

  /**
   * Compute person bounding box from the last mask.
   * Returns normalized coordinates (0-1) with padding, or null if no person detected.
   */
  getPersonBBox(padding = 0.15): CropRegion | null {
    const mask = this.fullMask ?? this.lastMask;
    if (!mask) return null;

    const mw = this.config.outputWidth;
    const mh = this.config.outputHeight;
    let minX = mw, minY = mh, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        if (mask[y * mw + x] > 0.5) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }

    if (!found) return null;

    // Normalize to 0-1 and add padding
    let nx = minX / mw - padding;
    let ny = minY / mh - padding;
    let nw = (maxX - minX) / mw + padding * 2;
    let nh = (maxY - minY) / mh + padding * 2;

    // Clamp to frame bounds
    nx = Math.max(0, nx);
    ny = Math.max(0, ny);
    nw = Math.min(nw, 1 - nx);
    nh = Math.min(nh, 1 - ny);

    return { x: nx, y: ny, w: nw, h: nh };
  }

  get maskWidth(): number {
    return this.config.outputWidth;
  }

  get maskHeight(): number {
    return this.config.outputHeight;
  }

  /** Release model resources */
  destroy(): void {
    this.segmenter?.close();
    this.segmenter = null;
    this.lastMask = null;
    this.previousMask = null;
    this.fullMask = null;
    this.lastCropRegion = null;
  }
}

