/**
 * Auto-Framing (Centering) Module
 *
 * Google Meet's "automatic framing" feature that keeps the subject centered
 * in frame. Works by deriving a bounding box from the segmentation mask
 * and applying a smoothed crop.
 *
 * Two modes:
 * 1. Mask-based: derives person bounding box from segmentation mask (no extra model)
 * 2. Face-based: uses face detection for more stable centering (optional, requires
 *    MediaPipe Face Detector)
 *
 * The crop is smoothed with a low-pass filter to avoid "camera swim."
 *
 * Behavior (matching Meet):
 * - On join: center once
 * - With virtual background: allow continuous gentle adjustments
 * - Without virtual background: center on join, then mostly static
 * - Always maintain headroom proportions
 */

export interface AutoFrameConfig {
  /** Enable auto-framing (default: false) */
  enabled?: boolean;
  /** Framing mode: 'mask' uses segmentation mask, 'face' uses face detection */
  mode?: 'mask' | 'face';
  /** How much headroom above the subject (fraction of height, default: 0.15) */
  headroom?: number;
  /** Minimum padding around subject (fraction of dimension, default: 0.1) */
  padding?: number;
  /** Smoothing factor for crop movement (0-1, higher = slower/smoother, default: 0.92) */
  smoothing?: number;
  /** Maximum zoom level (1.0 = no zoom, 2.0 = 2x, default: 1.5) */
  maxZoom?: number;
  /** Minimum zoom level (default: 1.0) */
  minZoom?: number;
  /** Whether to allow continuous adjustments (default: true when virtual bg enabled) */
  continuous?: boolean;
  /** Dead zone — don't adjust if subject moved less than this fraction (default: 0.03) */
  deadZone?: number;
}

export interface CropRect {
  /** X offset as fraction of frame width (0-1) */
  x: number;
  /** Y offset as fraction of frame height (0-1) */
  y: number;
  /** Crop width as fraction of frame width (0-1) */
  width: number;
  /** Crop height as fraction of frame height (0-1) */
  height: number;
  /** Computed zoom level */
  zoom: number;
}

interface BBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  area: number;
}

export class AutoFramer {
  private config: Required<AutoFrameConfig>;

  // Smoothed crop state
  private currentCrop: CropRect = { x: 0, y: 0, width: 1, height: 1, zoom: 1 };
  private targetCrop: CropRect = { x: 0, y: 0, width: 1, height: 1, zoom: 1 };

  // Tracking
  private frameCount = 0;
  private lastBBox: BBox | null = null;
  private hasInitialFrame = false;
  private lockedZoom: number | null = null; // Lock zoom after initial stabilization

  // Frame dimensions
  private frameWidth = 1280;
  private frameHeight = 720;
  private frameAspect = 16 / 9;

  constructor(config: AutoFrameConfig = {}) {
    this.config = {
      enabled: config.enabled ?? false,
      mode: config.mode ?? 'mask',
      headroom: config.headroom ?? 0.15,
      padding: config.padding ?? 0.1,
      smoothing: config.smoothing ?? 0.75,
      maxZoom: config.maxZoom ?? 1.5,
      minZoom: config.minZoom ?? 1.2,
      continuous: config.continuous ?? true,
      deadZone: config.deadZone ?? 0.03,
    };
  }

  /** Set frame dimensions */
  setFrameSize(width: number, height: number): void {
    this.frameWidth = width;
    this.frameHeight = height;
    this.frameAspect = width / height;
  }

  /**
   * Update framing from a segmentation mask.
   *
   * Call this every frame (or every model frame) with the current mask.
   * Returns the smoothed crop rectangle to apply.
   *
   * @param mask - Segmentation mask (Float32Array, values 0-1)
   * @param maskWidth - Width of the mask
   * @param maskHeight - Height of the mask
   * @returns Smoothed crop rectangle (fractions of frame dimensions)
   */
  updateFromMask(
    mask: Float32Array,
    maskWidth: number,
    maskHeight: number,
  ): CropRect {
    if (!this.config.enabled) {
      return { x: 0, y: 0, width: 1, height: 1, zoom: 1 };
    }

    // Derive bounding box from mask
    const bbox = this.computeMaskBBox(mask, maskWidth, maskHeight);

    if (!bbox || bbox.area < 0.01) {
      // No person detected — hold current crop
      return this.currentCrop;
    }

    // Compute target crop from bounding box
    this.targetCrop = this.computeCropFromBBox(bbox);

    // Apply smoothing
    this.smoothCrop();

    this.lastBBox = bbox;
    this.frameCount++;
    this.hasInitialFrame = true;

    return { ...this.currentCrop };
  }

  /**
   * Update framing from a face bounding box.
   * More stable than mask-based since faces are smaller and less noisy.
   */
  updateFromFace(
    faceX: number,  // center X as fraction of frame
    faceY: number,  // center Y as fraction of frame
    faceWidth: number,  // face width as fraction of frame
    faceHeight: number, // face height as fraction of frame
  ): CropRect {
    if (!this.config.enabled) {
      return { x: 0, y: 0, width: 1, height: 1, zoom: 1 };
    }

    // Convert face to approximate body bounding box
    // Head is roughly 1/7 of body height
    const bodyHeight = Math.min(faceHeight * 5, 0.95); // estimate body
    const bodyWidth = Math.min(faceWidth * 3, 0.8);

    const bbox: BBox = {
      centerX: faceX,
      centerY: faceY + faceHeight * 1.5, // body center is below face
      minX: faceX - bodyWidth / 2,
      maxX: faceX + bodyWidth / 2,
      minY: faceY - faceHeight * 0.5,
      maxY: faceY + bodyHeight,
      width: bodyWidth,
      height: bodyHeight,
      area: bodyWidth * bodyHeight,
    };

    this.targetCrop = this.computeCropFromBBox(bbox);
    this.smoothCrop();

    this.lastBBox = bbox;
    this.frameCount++;
    this.hasInitialFrame = true;

    return { ...this.currentCrop };
  }

  /** Reset to default (full frame, no crop) */
  reset(): void {
    this.currentCrop = { x: 0, y: 0, width: 1, height: 1, zoom: 1 };
    this.targetCrop = { x: 0, y: 0, width: 1, height: 1, zoom: 1 };
    this.lastBBox = null;
    this.hasInitialFrame = false;
    this.frameCount = 0;
    this.lockedZoom = null;
  }

  /** Get current crop without updating */
  getCurrentCrop(): CropRect {
    return { ...this.currentCrop };
  }

  /** Update configuration */
  updateConfig(config: Partial<AutoFrameConfig>): void {
    Object.assign(this.config, config);
  }

  // === Private ===

  private computeMaskBBox(
    mask: Float32Array,
    width: number,
    height: number,
  ): BBox | null {
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    let totalWeight = 0;
    let weightedX = 0;
    let weightedY = 0;

    const threshold = 0.5;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const val = mask[y * width + x];
        if (val > threshold) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          weightedX += x * val;
          weightedY += y * val;
          totalWeight += val;
        }
      }
    }

    if (totalWeight < 1) return null;

    // Normalize to 0-1 fractions
    const result: BBox = {
      minX: minX / width,
      minY: minY / height,
      maxX: maxX / width,
      maxY: maxY / height,
      centerX: (weightedX / totalWeight) / width,
      centerY: (weightedY / totalWeight) / height,
      width: (maxX - minX) / width,
      height: (maxY - minY) / height,
      area: 0,
    };
    result.area = result.width * result.height;

    return result;
  }

  private computeCropFromBBox(bbox: BBox): CropRect {
    const { maxZoom, minZoom } = this.config;

    // Target: person should fill ~90% of the output (largest dimension).
    // Zoom = targetFill / actualFill. Further away = more zoom, closer = less.
    // Person at 80% → 1.12x, at 60% → 1.5x (max), at 50% → 1.5x (capped).
    const targetFill = 0.9;
    const actualFill = Math.max(bbox.width, bbox.height);
    const rawZoom = actualFill > 0.01 ? targetFill / actualFill : 1.0;
    const zoom = Math.max(minZoom, Math.min(maxZoom, rawZoom));

    const cropSize = 1 / zoom;
    const cropW = cropSize;
    const cropH = cropSize;

    // Position person's bbox center in the lower portion of the crop.
    // Higher offset = person lower in crop = head higher in output.
    // The bbox center is typically at chest level, so we need offset > 0.5
    // to push the head into the upper third (like Google Meet).
    // Close (fills frame): 0.55 — head in upper third with headroom
    // Far (small in frame): 0.58 — head well above center
    const fillRatio = Math.max(bbox.width, bbox.height);
    const vertOffset = 0.55 + (1 - fillRatio) * 0.03;
    let cropX = bbox.centerX - cropW / 2;
    let cropY = bbox.centerY - cropH * vertOffset;

    // Clamp to frame bounds
    cropX = Math.max(0, Math.min(cropX, 1 - cropW));
    cropY = Math.max(0, Math.min(cropY, 1 - cropH));

    return {
      x: cropX,
      y: cropY,
      width: Math.min(cropW, 1),
      height: Math.min(cropH, 1),
      zoom,
    };
  }

  private smoothCrop(): void {
    const s = this.config.smoothing;

    // On first frame, snap immediately
    if (!this.hasInitialFrame) {
      this.currentCrop = { ...this.targetCrop };
      return;
    }

    // Dead zone removed — the smoothing (0.75) already prevents jitter.
    // The old 3% dead zone blocked horizontal centering for small movements.

    if (!this.config.continuous && this.frameCount > 30) {
      // Non-continuous mode: only center on join (first 30 frames)
      return;
    }

    // Exponential moving average (low-pass filter)
    this.currentCrop.x = this.lerp(this.currentCrop.x, this.targetCrop.x, 1 - s);
    this.currentCrop.y = this.lerp(this.currentCrop.y, this.targetCrop.y, 1 - s);
    this.currentCrop.width = this.lerp(this.currentCrop.width, this.targetCrop.width, 1 - s);
    this.currentCrop.height = this.lerp(this.currentCrop.height, this.targetCrop.height, 1 - s);
    this.currentCrop.zoom = this.lerp(this.currentCrop.zoom, this.targetCrop.zoom, 1 - s);
  }

  private lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
  }
}
