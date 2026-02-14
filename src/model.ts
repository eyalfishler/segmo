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
  private motionBuffer: Float32Array | null = null; // Reused buffer for motion map (avoids GC)
  // Motion compensation: 3-zone EMA-smoothed velocity for mask shift prediction
  private prevCentroid = [0.5, 0.5, 0.5]; // [topX, midX, botX] normalized
  private velocity = [0, 0, 0]; // EMA-smoothed X velocity per Y zone
  private velocityY = 0; // EMA-smoothed Y velocity (whole body)
  private prevCentroidYAll = 0.5;
  private hasPreviousMask = false; // True after first real frame (prevents init spike)
  private firstCentroid = true; // Seed centroid from first detection
  // Cached bbox from last segment() call — computed during ROI mapping or mask extraction
  private cachedBBoxMinX = 0;
  private cachedBBoxMinY = 0;
  private cachedBBoxMaxX = 0;
  private cachedBBoxMaxY = 0;
  private cachedBBoxFound = false;

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
        // Save previous mask in full-frame space (fullMask still holds last frame's result)
        const prev = this.fullMask ?? this.lastMask;
        if (prev) {
          this.previousMask!.set(prev);
          this.hasPreviousMask = true;
        }
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
    // Optimized: only iterate within crop bounds (skips 30-50% of pixels)
    // and compute person bbox during the same pass (eliminates separate 65K scan)
    if (this.lastCropRegion && this.lastMask) {
      const mw = outputWidth;
      const mh = outputHeight;
      const totalPixels = mw * mh;

      if (!this.fullMask || this.fullMask.length !== totalPixels) {
        this.fullMask = new Float32Array(totalPixels);
      }
      this.fullMask.fill(0); // background outside crop

      const crop = this.lastCropRegion;
      const cx0 = crop.x * mw;
      const cy0 = crop.y * mh;
      const cw = crop.w * mw;
      const ch = crop.h * mh;

      // Iterate only within crop bounds (major perf win)
      const x0 = Math.max(0, Math.floor(cx0));
      const y0 = Math.max(0, Math.floor(cy0));
      const x1 = Math.min(mw, Math.ceil(cx0 + cw));
      const y1 = Math.min(mh, Math.ceil(cy0 + ch));

      // Precompute scale factors — maps full-frame pixel to crop-mask pixel directly
      // Original: sx = floor((x - cx0) / cw * mw) = floor((x - cx0) * mw / cw)
      const scaleX = mw / cw; // combines /cw and *mw into one multiply
      const scaleY = mh / ch;
      const mwMinus1 = mw - 1;
      const mhMinus1 = mh - 1;

      // Track bbox during mapping (eliminates separate getPersonBBox scan)
      let bMinX = mw, bMinY = mh, bMaxX = 0, bMaxY = 0;
      let bFound = false;

      for (let y = y0; y < y1; y++) {
        const sy = Math.min(((y - cy0) * scaleY) | 0, mhMinus1);
        const yOff = y * mw;
        const syOff = sy * mw;

        for (let x = x0; x < x1; x++) {
          const sx = Math.min(((x - cx0) * scaleX) | 0, mwMinus1);
          const val = this.lastMask[syOff + sx];
          this.fullMask[yOff + x] = val;

          // Track person bbox during same pass
          if (val > 0.5) {
            if (x < bMinX) bMinX = x;
            if (x > bMaxX) bMaxX = x;
            if (y < bMinY) bMinY = y;
            if (y > bMaxY) bMaxY = y;
            bFound = true;
          }
        }
      }

      this.cachedBBoxMinX = bMinX;
      this.cachedBBoxMinY = bMinY;
      this.cachedBBoxMaxX = bMaxX;
      this.cachedBBoxMaxY = bMaxY;
      this.cachedBBoxFound = bFound;
      if (bFound) this.updateCentroidMotion(bMinX, bMinY, bMaxX, bMaxY);

      return this.fullMask;
    }

    // Non-ROI path: compute bbox during mask (already copied above)
    this.computeBBoxFromMask();
    if (this.cachedBBoxFound) {
      this.updateCentroidMotion(
        this.cachedBBoxMinX, this.cachedBBoxMinY,
        this.cachedBBoxMaxX, this.cachedBBoxMaxY,
      );
    }
    return this.lastMask;
  }

  /** Get the last computed mask (for interpolated frames) */
  getLastMask(): Float32Array | null {
    return this.fullMask ?? this.lastMask;
  }

  /**
   * Get per-pixel motion estimate between last two masks.
   * Reuses internal buffer to avoid allocating 65K floats every frame.
   */
  getMotionMap(): Float32Array | null {
    const current = this.fullMask ?? this.lastMask;
    if (!current || !this.previousMask || !this.hasPreviousMask) return null;

    // Reuse buffer (avoids GC pressure from allocating every frame)
    if (!this.motionBuffer || this.motionBuffer.length !== current.length) {
      this.motionBuffer = new Float32Array(current.length);
    }
    for (let i = 0; i < current.length; i++) {
      this.motionBuffer[i] = Math.abs(current[i] - this.previousMask[i]);
    }
    return this.motionBuffer;
  }

  /** Get 3-zone EMA-smoothed velocity for mask motion compensation */
  getMaskMotionVector(): { vx: [number, number, number]; vy: number } {
    return {
      vx: [this.velocity[0], this.velocity[1], this.velocity[2]],
      vy: this.velocityY,
    };
  }

  /** Update 3-zone centroid tracking from mask. Scans top/mid/bottom Y bands. */
  private updateCentroidMotion(_minX: number, minY: number, _maxX: number, maxY: number): void {
    const mask = this.fullMask ?? this.lastMask;
    if (!mask) return;
    const mw = this.config.outputWidth;
    const mh = this.config.outputHeight;
    const personH = maxY - minY;
    if (personH < 3) return;

    const bandH = personH / 3;
    const band1Start = minY + bandH;
    const band2Start = minY + 2 * bandH;

    const sumX = [0, 0, 0];
    const count = [0, 0, 0];
    let sumY = 0, countY = 0;

    for (let y = minY; y <= maxY; y++) {
      const off = y * mw;
      const band = y >= band2Start ? 2 : y >= band1Start ? 1 : 0;
      for (let x = 0; x < mw; x++) {
        if (mask[off + x] > 0.5) {
          sumX[band] += x;
          count[band]++;
          sumY += y;
          countY++;
        }
      }
    }

    const cx = [
      count[0] > 0 ? sumX[0] / count[0] / mw : 0.5,
      count[1] > 0 ? sumX[1] / count[1] / mw : 0.5,
      count[2] > 0 ? sumX[2] / count[2] / mw : 0.5,
    ];
    const cyAll = countY > 0 ? sumY / countY / mh : 0.5;

    // First detection: seed centroid without computing velocity (prevents init spike)
    if (this.firstCentroid) {
      this.prevCentroid = [cx[0], cx[1], cx[2]];
      this.prevCentroidYAll = cyAll;
      this.firstCentroid = false;
      return;
    }

    // EMA smoothing (α=0.8) — responsive to real movement, dampens noise
    const alpha = 0.8;
    for (let i = 0; i < 3; i++) {
      const rawV = cx[i] - this.prevCentroid[i];
      this.velocity[i] = alpha * rawV + (1 - alpha) * this.velocity[i];
      this.prevCentroid[i] = cx[i];
    }
    const rawVy = cyAll - this.prevCentroidYAll;
    this.velocityY = alpha * rawVy + (1 - alpha) * this.velocityY;
    this.prevCentroidYAll = cyAll;
  }

  /**
   * Get person bounding box from the last segment() call.
   * Uses cached bbox computed during ROI mapping or mask extraction —
   * no additional 65K pixel scan needed.
   */
  getPersonBBox(padding = 0.15): CropRegion | null {
    if (!this.cachedBBoxFound) return null;

    const mw = this.config.outputWidth;
    const mh = this.config.outputHeight;

    // Normalize cached pixel coords to 0-1 and add padding
    let nx = this.cachedBBoxMinX / mw - padding;
    let ny = this.cachedBBoxMinY / mh - padding;
    let nw = (this.cachedBBoxMaxX - this.cachedBBoxMinX) / mw + padding * 2;
    let nh = (this.cachedBBoxMaxY - this.cachedBBoxMinY) / mh + padding * 2;

    // Clamp to frame bounds
    nx = Math.max(0, nx);
    ny = Math.max(0, ny);
    nw = Math.min(nw, 1 - nx);
    nh = Math.min(nh, 1 - ny);

    return { x: nx, y: ny, w: nw, h: nh };
  }

  /** Compute bbox from lastMask (non-ROI path). Called once per segment(). */
  private computeBBoxFromMask(): void {
    const mask = this.lastMask;
    if (!mask) { this.cachedBBoxFound = false; return; }

    const mw = this.config.outputWidth;
    const mh = this.config.outputHeight;
    let minX = mw, minY = mh, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < mh; y++) {
      const yOff = y * mw;
      for (let x = 0; x < mw; x++) {
        if (mask[yOff + x] > 0.5) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }

    this.cachedBBoxMinX = minX;
    this.cachedBBoxMinY = minY;
    this.cachedBBoxMaxX = maxX;
    this.cachedBBoxMaxY = maxY;
    this.cachedBBoxFound = found;
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
    this.motionBuffer = null;
    this.lastCropRegion = null;
    this.cachedBBoxFound = false;
    this.hasPreviousMask = false;
    this.firstCentroid = true;
  }
}

