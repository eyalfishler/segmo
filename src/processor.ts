/**
 * LiveKit Background Segmentation Processor
 *
 * Drop-in replacement for @livekit/track-processors-js BackgroundProcessor.
 * Implements the same interface but with Google Meet-quality output.
 *
 * Architecture:
 * ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────────┐
 * │ Camera Frame │───▶│  Model (15fps)   │───▶│  WebGL Post-Processing   │
 * │  (30fps)     │    │  MediaPipe GPU   │    │  - Temporal Smooth       │
 * │              │───▶│  (skip frames)   │    │  - Bilateral Upsample    │
 * │              │    └──────────────────┘    │  - Edge Feather          │
 * │              │───────────────────────────▶│  - Composite             │
 * └─────────────┘                             └──────────────────────────┘
 *                                                        │
 *                                              ┌─────────▼─────────┐
 *                                              │  Output VideoFrame │
 *                                              │  (30fps smooth)    │
 *                                              └───────────────────┘
 *
 * Key performance tricks:
 * 1. Model runs at 15fps, frames are interpolated between model runs
 * 2. All post-processing is on GPU (WebGL2)
 * 3. Bilateral upsample uses camera frame as edge guide
 * 4. Temporal smoothing with hysteresis prevents flickering
 * 5. Background blur at half resolution with multi-pass Gaussian
 */

import { PostProcessingPipeline, type PipelineOptions } from './pipeline';
import { SegmentationModel, type ModelConfig } from './model';
import { AdaptiveQualityController, type AdaptiveConfig, type QualityLevel } from './adaptive';
import { AutoFramer, type AutoFrameConfig, type CropRect } from './autoframe';
import { ModelWorkerClient } from './model-worker';

export type BackgroundMode = 'blur' | 'image' | 'color' | 'none';

export interface SegmentationProcessorOptions {
  /** Background mode */
  backgroundMode?: BackgroundMode;
  /** Background blur radius (default: 12) */
  blurRadius?: number;
  /** Background color hex (default: '#00FF00') */
  backgroundColor?: string;
  /** Background image element */
  backgroundImage?: HTMLImageElement | null;
  /** Target model FPS — model runs at this rate, display interpolates (default: 15) */
  modelFps?: number;
  /** Target output FPS (default: 30) */
  outputFps?: number;
  /** Model configuration overrides */
  modelConfig?: ModelConfig;
  /** Pipeline quality presets */
  quality?: 'low' | 'medium' | 'high';
  /** Enable performance metrics logging */
  debug?: boolean;
  /** Enable adaptive quality (auto-adjusts to device performance, default: true) */
  adaptive?: boolean;
  /** Adaptive quality configuration overrides */
  adaptiveConfig?: AdaptiveConfig;
  /** Auto-framing configuration */
  autoFrame?: AutoFrameConfig;
  /** Run model inference in a Web Worker (frees main thread, default: false) */
  useWorker?: boolean;
}

// Quality presets tuned by hand
const QUALITY_PRESETS = {
  low: {
    appearRate: 0.85,
    disappearRate: 0.45,
    featherRadius: 2.0,
    rangeSigma: 0.15,
    lightWrap: false,
    morphology: false,
    blurRadius: 8,
    modelWidth: 160,
    modelHeight: 160,
    modelFps: 10,
  },
  medium: {
    appearRate: 0.75,
    disappearRate: 0.35,
    featherRadius: 3.0,
    rangeSigma: 0.1,
    lightWrap: true,
    morphology: true,
    blurRadius: 12,
    modelWidth: 256,
    modelHeight: 256,
    modelFps: 15,
  },
  high: {
    appearRate: 0.7,
    disappearRate: 0.3,
    featherRadius: 4.0,
    rangeSigma: 0.08,
    lightWrap: true,
    morphology: true,
    blurRadius: 16,
    modelWidth: 256,
    modelHeight: 256,
    modelFps: 30,
  },
};

interface PerformanceMetrics {
  modelInferenceMs: number;
  pipelineMs: number;
  totalFrameMs: number;
  fps: number;
  modelFps: number;
  skippedFrames: number;
}

export class SegmentationProcessor {
  private pipeline: PostProcessingPipeline | null = null;
  private model: SegmentationModel | null = null;
  private adaptive: AdaptiveQualityController | null = null;
  private autoFramer: AutoFramer;
  private opts: Required<SegmentationProcessorOptions>;
  private qualityPreset: typeof QUALITY_PRESETS.medium;

  // Frame scheduling
  private lastModelTime = 0;
  private modelInterval: number;
  private frameCount = 0;
  private skippedFrames = 0;

  // Performance tracking
  private metrics: PerformanceMetrics = {
    modelInferenceMs: 0,
    pipelineMs: 0,
    totalFrameMs: 0,
    fps: 0,
    modelFps: 0,
    skippedFrames: 0,
  };
  private fpsCounter = 0;
  private fpsTimestamp = 0;
  private modelFpsCounter = 0;

  // State
  private initialized = false;
  private width = 0;
  private height = 0;

  // ROI cropping: use previous frame's person bbox to crop next frame's model input
  private personCropRegion: import('./model').CropRegion | null = null;

  // Web Worker for off-main-thread inference
  private workerClient: ModelWorkerClient | null = null;
  private workerMask: Float32Array | null = null;
  private workerMotion: Float32Array | null = null;
  private workerBBox: { minX: number; minY: number; maxX: number; maxY: number } | null = null;
  private workerHasFreshMask = false;

  // Mask motion compensation: 3-zone velocity prediction
  private maskVx: [number, number, number] = [0, 0, 0];
  private maskVy = 0;
  private interpFrameCount = 0;

  constructor(options: SegmentationProcessorOptions = {}) {
    this.opts = {
      backgroundMode: 'blur',
      blurRadius: 12,
      backgroundColor: '#00FF00',
      backgroundImage: null,
      modelFps: 15,
      outputFps: 30,
      modelConfig: {},
      quality: 'medium',
      debug: false,
      adaptive: true,
      adaptiveConfig: {},
      autoFrame: {},
      useWorker: false,
      ...options,
    };

    this.qualityPreset = QUALITY_PRESETS[this.opts.quality];
    this.modelInterval = 1000 / (this.opts.modelFps || this.qualityPreset.modelFps);

    // Initialize auto-framer
    this.autoFramer = new AutoFramer(this.opts.autoFrame);

    // Initialize adaptive quality controller
    if (this.opts.adaptive) {
      this.adaptive = new AdaptiveQualityController({
        debug: this.opts.debug,
        ...this.opts.adaptiveConfig,
      });

      this.adaptive.onApply((level: QualityLevel) => {
        this.modelInterval = 1000 / level.modelFps;
        this.pipeline?.updateOptions({
          appearRate: level.appearRate,
          disappearRate: level.disappearRate,
          featherRadius: level.featherRadius,
          rangeSigma: level.rangeSigma,
          blurRadius: level.blurRadius,
        });
      });
    }
  }

  /**
   * Initialize the processor. Call once before processing frames.
   * Can be called during LiveKit track processor init.
   */
  async init(width: number, height: number): Promise<void> {
    // Clean up previous init (e.g., camera switch triggers re-init)
    if (this.workerClient) {
      this.workerClient.destroy();
      this.workerClient = null;
    }
    if (this.pipeline) {
      this.pipeline.destroy();
      this.pipeline = null;
    }
    if (this.model) {
      this.model.destroy();
      this.model = null;
    }

    this.width = width;
    this.height = height;

    const preset = this.qualityPreset;

    // Initialize model (skip MediaPipe load when worker handles inference)
    this.model = new SegmentationModel({
      outputWidth: preset.modelWidth,
      outputHeight: preset.modelHeight,
      delegate: 'CPU',
      ...this.opts.modelConfig,
    });
    if (!this.opts.useWorker) {
      await this.model.init();
    }

    // Initialize WebGL pipeline
    this.pipeline = new PostProcessingPipeline({
      width,
      height,
      maskWidth: this.model.maskWidth,
      maskHeight: this.model.maskHeight,
      backgroundMode: this.opts.backgroundMode === 'none' ? 'blur' : this.opts.backgroundMode,
      backgroundColor: this.opts.backgroundColor,
      backgroundImage: this.opts.backgroundImage,
      blurRadius: this.opts.blurRadius || preset.blurRadius,
      lightWrap: preset.lightWrap,
      morphology: preset.morphology,
      appearRate: preset.appearRate,
      disappearRate: preset.disappearRate,
      featherRadius: preset.featherRadius,
      rangeSigma: preset.rangeSigma,
    });

    this.initialized = true;

    // Configure auto-framer with actual dimensions
    this.autoFramer.setFrameSize(width, height);
    this.log('Initialized', {
      resolution: `${width}x${height}`,
      modelResolution: `${this.model.maskWidth}x${this.model.maskHeight}`,
      modelFps: this.opts.modelFps || preset.modelFps,
      quality: this.opts.quality,
      adaptive: this.opts.adaptive,
    });

    // Run quick benchmark for adaptive quality calibration
    if (this.adaptive) {
      this.log('Running initial calibration benchmark...');
      // The adaptive controller will calibrate after the first ~30 real frames,
      // but we can seed it based on the quality preset
      const presetToTier: Record<string, number> = {
        high: 0,
        medium: 1,
        low: 3,
      };
      this.adaptive.setTier(presetToTier[this.opts.quality] ?? 1);
      this.adaptive.unlock(); // Allow auto-adjustment after initial setting
    }

    // Initialize Web Worker for off-main-thread inference (optional)
    if (this.opts.useWorker) {
      try {
        this.workerClient = new ModelWorkerClient({
          outputWidth: preset.modelWidth,
          outputHeight: preset.modelHeight,
          delegate: 'CPU',
          ...this.opts.modelConfig,
        });
        this.workerClient.onMaskReady((result) => {
          this.workerMask = result.mask;
          this.workerMotion = result.motion;
          this.workerBBox = result.bbox;
          this.workerHasFreshMask = true;
        });
        await this.workerClient.init();
        this.log('Worker initialized — inference runs off main thread');
      } catch (e) {
        this.log('Worker init failed, falling back to main thread', { error: String(e) });
        this.workerClient?.destroy();
        this.workerClient = null;
      }
    }
  }

  /**
   * Process a single video frame.
   *
   * This is the main entry point called for every camera frame.
   * Decides whether to run the model or interpolate, then runs post-processing.
   *
   * @param frame - Input camera frame
   * @param timestamp - Frame timestamp (performance.now())
   * @returns Processed output canvas, or null to pass through original
   */
  processFrame(frame: TexImageSource, timestamp: number): OffscreenCanvas | null {
    if (!this.initialized || !this.pipeline || !this.model) return null;
    if (this.opts.backgroundMode === 'none') return null;

    const frameStart = performance.now();
    this.updateFpsCounter(timestamp);

    // Adaptive model rate: run model faster during motion to reduce mask lag.
    // Up to 4x speedup, capped at display refresh (~60fps / 16ms min).
    const maxVx = Math.max(Math.abs(this.maskVx[0]), Math.abs(this.maskVx[1]), Math.abs(this.maskVx[2]));
    const motionMag = Math.sqrt(maxVx * maxVx + this.maskVy * this.maskVy);
    const motionSpeedup = Math.min(4.0, 1.0 + motionMag * 20);
    const effectiveInterval = Math.max(16, this.modelInterval / motionSpeedup);
    const timeSinceLastModel = timestamp - this.lastModelTime;
    const shouldRunModel = timeSinceLastModel >= effectiveInterval;

    // Apply auto-frame crop BEFORE rendering so centering is immediate
    const cropRect = this.autoFramer.getCurrentCrop();
    if (cropRect.zoom > 1.02) {
      this.pipeline.setCropRect({
        x: cropRect.x,
        y: cropRect.y,
        w: cropRect.width,
        h: cropRect.height,
      });
      // if (this.frameCount % 60 === 0) {
      //   console.log(`[AutoFrame] crop x=${cropRect.x.toFixed(3)} y=${cropRect.y.toFixed(3)} w=${cropRect.width.toFixed(3)} h=${cropRect.height.toFixed(3)} zoom=${cropRect.zoom.toFixed(3)}`);
      // }
    } else {
      this.pipeline.setCropRect(null);
      // if (this.frameCount % 60 === 0) {
      //   console.log(`[AutoFrame] NO CROP zoom=${cropRect.zoom.toFixed(3)}`);
      // }
    }

    let output: OffscreenCanvas;

    // --- Worker path: non-blocking inference off main thread ---
    if (this.workerClient) {
      if (this.workerHasFreshMask && this.workerMask) {
        this.workerHasFreshMask = false;
        this.modelFpsCounter++;
        this.updateROICrop(this.workerBBox);
        this.autoFramer.updateFromMask(
          this.workerMask, this.model.maskWidth, this.model.maskHeight,
        );
        // Capture motion vector + reset interpolation counter
        const mv = this.model.getMaskMotionVector();
        this.maskVx = mv.vx;
        this.maskVy = mv.vy;
        this.interpFrameCount = 0;

        // const vxW = mv.vx[0] * 0.6 + mv.vx[1] * 0.3 + mv.vx[2] * 0.1;
        // if (Math.abs(vxW) > 0.0005 || Math.abs(mv.vy) > 0.0005) {
        //   console.log(`[Motion] vx=[${mv.vx[0].toFixed(4)}, ${mv.vx[1].toFixed(4)}, ${mv.vx[2].toFixed(4)}] vy=${mv.vy.toFixed(4)} | model frame`);
        // }

        const pipelineStart = performance.now();
        output = this.pipeline.process(frame, this.workerMask, this.workerMotion);
        this.metrics.pipelineMs = performance.now() - pipelineStart;
        this.metrics.modelInferenceMs = 0;
      } else {
        // Interpolate with accumulating motion-compensated shift
        this.skippedFrames++;
        this.interpFrameCount++;
        const pipelineStart = performance.now();
        output = this.pipeline.processInterpolated(frame, this.getAccumulatedShift());
        this.metrics.pipelineMs = performance.now() - pipelineStart;
      }

      if (shouldRunModel) {
        this.workerClient.requestSegment(frame, timestamp, this.personCropRegion);
        this.lastModelTime = timestamp;
      }
    }
    // --- Main thread path: blocking inference ---
    else if (shouldRunModel) {
      const modelStart = performance.now();
      const mask = this.model.segment(
        frame, timestamp,
        this.personCropRegion,
        this.width, this.height,
      );
      this.metrics.modelInferenceMs = performance.now() - modelStart;

      this.lastModelTime = timestamp;
      this.modelFpsCounter++;

      if (mask) {
        const afZoom = this.autoFramer.getCurrentCrop().zoom;
        const roiPadding = afZoom > 1.02 ? 0.05 * afZoom : 0.05;
        const rawBBox = this.model.getPersonBBox(roiPadding);
        this.updateROICropFromBBox(rawBBox);
        this.autoFramer.updateFromMask(
          mask, this.model.maskWidth, this.model.maskHeight,
        );
        const motionMap = this.model.getMotionMap();

        // Capture motion vector + reset interpolation counter
        const mv = this.model.getMaskMotionVector();
        this.maskVx = mv.vx;
        this.maskVy = mv.vy;
        this.interpFrameCount = 0;

        // const vxW = mv.vx[0] * 0.6 + mv.vx[1] * 0.3 + mv.vx[2] * 0.1;
        // if (Math.abs(vxW) > 0.0005 || Math.abs(mv.vy) > 0.0005) {
        //   console.log(`[Motion] vx=[${mv.vx[0].toFixed(4)}, ${mv.vx[1].toFixed(4)}, ${mv.vx[2].toFixed(4)}] vy=${mv.vy.toFixed(4)} | model frame`);
        // }

        const pipelineStart = performance.now();
        output = this.pipeline.process(frame, mask, motionMap);
        this.metrics.pipelineMs = performance.now() - pipelineStart;
      } else {
        this.interpFrameCount++;
        const pipelineStart = performance.now();
        output = this.pipeline.processInterpolated(frame, this.getAccumulatedShift());
        this.metrics.pipelineMs = performance.now() - pipelineStart;
      }
    } else {
      // Not time for model — interpolate with motion-compensated mask
      this.skippedFrames++;
      this.interpFrameCount++;
      const pipelineStart = performance.now();
      output = this.pipeline.processInterpolated(frame, this.getAccumulatedShift());
      this.metrics.pipelineMs = performance.now() - pipelineStart;
    }

    this.metrics.totalFrameMs = performance.now() - frameStart;
    this.frameCount++;

    // Feed frame time to adaptive quality controller
    if (this.adaptive) {
      this.adaptive.reportFrame(this.metrics.totalFrameMs);
    }

    if (this.opts.debug && this.frameCount % 60 === 0) {
      this.logMetrics();
    }

    return output;
  }

  /**
   * LiveKit TrackProcessor-compatible interface.
   *
   * Implements the official processedTrack pattern from
   * https://github.com/livekit/track-processors-js
   *
   * ```ts
   * const processor = new SegmentationProcessor({ backgroundMode: 'blur' });
   * const track = await createLocalVideoTrack();
   * await track.setProcessor(processor.toLiveKitProcessor());
   * ```
   */
  toLiveKitProcessor(): {
    name: string;
    init: (opts: { track: MediaStreamTrack; kind: string; element?: HTMLMediaElement }) => Promise<void>;
    restart: (opts: { track: MediaStreamTrack; kind: string; element?: HTMLMediaElement }) => Promise<void>;
    destroy: () => Promise<void>;
    processedTrack?: MediaStreamTrack;
  } {
    // Pipeline state owned by the processor object
    let pipelineAbort: AbortController | null = null;
    let generator: MediaStreamTrackGenerator | null = null;

    const startPipeline = (track: MediaStreamTrack, result: { processedTrack?: MediaStreamTrack }) => {
      // Create Insertable Streams pipeline: input → transform → output
      const trackProcessor = new MediaStreamTrackProcessor({ track });
      generator = new MediaStreamTrackGenerator({ kind: 'video' });
      pipelineAbort = new AbortController();

      const transformer = new TransformStream<VideoFrame, VideoFrame>({
        transform: (frame, controller) => {
          const timestamp = frame.timestamp ?? performance.now();
          const output = this.processFrame(frame, timestamp / 1000);

          if (output) {
            const outputFrame = new VideoFrame(output, {
              timestamp: frame.timestamp,
              alpha: 'discard',
            });
            frame.close();
            controller.enqueue(outputFrame);
          } else {
            controller.enqueue(frame);
          }
        },
      });

      trackProcessor.readable
        .pipeThrough(transformer, { signal: pipelineAbort.signal })
        .pipeTo(generator.writable, { signal: pipelineAbort.signal })
        .catch(() => { /* aborted — expected on destroy/restart */ });

      result.processedTrack = generator as unknown as MediaStreamTrack;
    };

    const stopPipeline = () => {
      pipelineAbort?.abort();
      pipelineAbort = null;
      generator = null;
    };

    const result: {
      name: string;
      processedTrack?: MediaStreamTrack;
      init: (opts: { track: MediaStreamTrack; kind: string; element?: HTMLMediaElement }) => Promise<void>;
      restart: (opts: { track: MediaStreamTrack; kind: string; element?: HTMLMediaElement }) => Promise<void>;
      destroy: () => Promise<void>;
    } = {
      name: 'segmo-segmentation',
      processedTrack: undefined,

      init: async (opts) => {
        const settings = opts.track.getSettings();
        const width = settings.width || 1280;
        const height = settings.height || 720;
        await this.init(width, height);
        startPipeline(opts.track, result);
      },

      restart: async (opts) => {
        stopPipeline();
        const settings = opts.track.getSettings();
        const width = settings.width || 1280;
        const height = settings.height || 720;
        if (width !== this.width || height !== this.height) {
          await this.init(width, height);
        }
        startPipeline(opts.track, result);
      },

      destroy: async () => {
        stopPipeline();
        this.destroy();
      },
    };

    return result;
  }

  /**
   * Create a MediaStream transformer for use outside LiveKit.
   *
   * ```ts
   * const processor = new SegmentationProcessor({ backgroundMode: 'blur' });
   * const outputTrack = await processor.createProcessedTrack(inputVideoTrack);
   * ```
   */
  async createProcessedTrack(inputTrack: MediaStreamTrack): Promise<MediaStreamTrack> {
    const settings = inputTrack.getSettings();
    await this.init(settings.width || 1280, settings.height || 720);

    // Use Insertable Streams (WebCodecs) if available
    const trackProcessor = new MediaStreamTrackProcessor({ track: inputTrack });
    const trackGenerator = new MediaStreamTrackGenerator({ kind: 'video' });

    const transformer = new TransformStream<VideoFrame, VideoFrame>({
      transform: (frame, controller) => {
        const timestamp = frame.timestamp ?? performance.now();
        const output = this.processFrame(frame, timestamp / 1000);

        if (output) {
          const outputFrame = new VideoFrame(output, {
            timestamp: frame.timestamp,
            alpha: 'discard',
          });
          frame.close();
          controller.enqueue(outputFrame);
        } else {
          controller.enqueue(frame);
        }
      },
    });

    trackProcessor.readable
      .pipeThrough(transformer)
      .pipeTo(trackGenerator.writable);

    return trackGenerator as unknown as MediaStreamTrack;
  }

  // === Runtime configuration ===

  /** Change background mode */
  setBackgroundMode(mode: BackgroundMode): void {
    this.opts.backgroundMode = mode;
    if (mode !== 'none' && this.pipeline) {
      this.pipeline.updateOptions({ backgroundMode: mode });
    }
  }

  /** Set background color (hex string) */
  setBackgroundColor(color: string): void {
    this.opts.backgroundColor = color;
    this.pipeline?.updateOptions({ backgroundColor: color });
  }

  /** Set background image */
  setBackgroundImage(image: HTMLImageElement): void {
    this.opts.backgroundImage = image;
    this.pipeline?.updateOptions({ backgroundImage: image, backgroundMode: 'image' });
  }

  /** Set blur radius */
  setBlurRadius(radius: number): void {
    this.opts.blurRadius = radius;
    this.pipeline?.updateOptions({ blurRadius: radius });
  }

  /** Change quality preset */
  setQuality(quality: 'low' | 'medium' | 'high'): void {
    this.opts.quality = quality;
    this.qualityPreset = QUALITY_PRESETS[quality];
    this.modelInterval = 1000 / this.qualityPreset.modelFps;

    if (this.pipeline) {
      this.pipeline.updateOptions({
        appearRate: this.qualityPreset.appearRate,
        disappearRate: this.qualityPreset.disappearRate,
        featherRadius: this.qualityPreset.featherRadius,
        rangeSigma: this.qualityPreset.rangeSigma,
      });
    }
  }

  /** Get current performance metrics */
  getMetrics(): Readonly<PerformanceMetrics> {
    return { ...this.metrics };
  }

  /** Get the adaptive quality controller (for monitoring/overriding) */
  getAdaptiveController(): AdaptiveQualityController | null {
    return this.adaptive;
  }

  /** Get the auto-framer (for configuration) */
  getAutoFramer(): AutoFramer {
    return this.autoFramer;
  }

  /**
   * Get current auto-frame crop rectangle.
   * Use this to apply the crop in your rendering layer.
   *
   * Returns fractions of frame dimensions:
   * { x: 0.1, y: 0.05, width: 0.8, height: 0.9, zoom: 1.2 }
   */
  getCropRect(): CropRect {
    return this.autoFramer.getCurrentCrop();
  }

  /** Get the current ROI crop region used for model input (debug) */
  getModelCropRegion(): { x: number; y: number; w: number; h: number } | null {
    return this.personCropRegion;
  }

  /** Enable/disable auto-framing */
  setAutoFrame(enabled: boolean, continuous?: boolean): void {
    this.autoFramer.updateConfig({
      enabled,
      continuous: continuous ?? (this.opts.backgroundMode !== 'none'),
    });
    if (!enabled) this.autoFramer.reset();
  }

  /** Clean up all resources */
  destroy(): void {
    this.pipeline?.destroy();
    this.model?.destroy();
    this.workerClient?.destroy();
    this.pipeline = null;
    this.model = null;
    this.adaptive = null;
    this.workerClient = null;
    this.workerMask = null;
    this.workerMotion = null;
    this.initialized = false;
  }

  // === Private helpers ===

  /** 3-zone constant-velocity shift, weighted by zone importance.
   * Head zone 60%, mid 30%, bottom 10%.
   * Capped to ±0.12 — seated person's realistic range. */
  private getAccumulatedShift(): { dx: number; dy: number } {
    const t = this.interpFrameCount;
    const amp = 1.0; // velocities are in full-frame normalized space (~0.01 during movement)
    const vx = this.maskVx[0] * 0.6 + this.maskVx[1] * 0.3 + this.maskVx[2] * 0.1;
    let dx = vx * t * amp;
    let dy = this.maskVy * t * amp;
    const maxShift = 0.12;
    dx = Math.max(-maxShift, Math.min(maxShift, dx));
    dy = Math.max(-maxShift, Math.min(maxShift, dy));

    // // Debug: log every frame when moving
    // if (Math.abs(vx) > 0.0005 || Math.abs(this.maskVy) > 0.0005) {
    //   console.log(`[Motion] vx=[${this.maskVx[0].toFixed(4)}, ${this.maskVx[1].toFixed(4)}, ${this.maskVx[2].toFixed(4)}] vy=${this.maskVy.toFixed(4)} | t=${t} shift=(${dx.toFixed(4)}, ${dy.toFixed(4)})`);
    // }

    return { dx, dy };
  }

  /** Update ROI crop from worker bbox (pixel coords from worker) */
  private updateROICrop(bbox: { minX: number; minY: number; maxX: number; maxY: number } | null): void {
    if (!bbox || !this.model) return;
    const mw = this.model.maskWidth;
    const mh = this.model.maskHeight;
    const afZoom = this.autoFramer.getCurrentCrop().zoom;
    const padding = afZoom > 1.02 ? 0.05 * afZoom : 0.05;

    let nx = bbox.minX / mw - padding;
    let ny = bbox.minY / mh - padding;
    let nw = (bbox.maxX - bbox.minX) / mw + padding * 2;
    let nh = (bbox.maxY - bbox.minY) / mh + padding * 2;
    nx = Math.max(0, nx);
    ny = Math.max(0, ny);
    nw = Math.min(nw, 1 - nx);
    nh = Math.min(nh, 1 - ny);

    this.updateROICropFromBBox({ x: nx, y: ny, w: nw, h: nh });
  }

  /** Apply dead-zone smoothing to a candidate ROI crop */
  private updateROICropFromBBox(rawBBox: import('./model').CropRegion | null): void {
    if (!rawBBox) return;
    if (this.personCropRegion) {
      const posShift = Math.max(
        Math.abs(rawBBox.x - this.personCropRegion.x),
        Math.abs(rawBBox.y - this.personCropRegion.y),
      );
      const sizeShift = Math.max(
        Math.abs(rawBBox.w - this.personCropRegion.w),
        Math.abs(rawBBox.h - this.personCropRegion.h),
      );
      const posChanged = posShift > 0.03;
      const sizeChanged = sizeShift > 0.015;

      if (posChanged || sizeChanged) {
        const s = 0.5;
        this.personCropRegion = {
          x: posChanged ? this.personCropRegion.x * s + rawBBox.x * (1 - s) : this.personCropRegion.x,
          y: posChanged ? this.personCropRegion.y * s + rawBBox.y * (1 - s) : this.personCropRegion.y,
          w: sizeChanged ? this.personCropRegion.w * s + rawBBox.w * (1 - s) : this.personCropRegion.w,
          h: sizeChanged ? this.personCropRegion.h * s + rawBBox.h * (1 - s) : this.personCropRegion.h,
        };
      }
    } else {
      this.personCropRegion = rawBBox;
    }
  }

  private updateFpsCounter(timestamp: number): void {
    this.fpsCounter++;
    if (timestamp - this.fpsTimestamp >= 1000) {
      this.metrics.fps = this.fpsCounter;
      this.metrics.modelFps = this.modelFpsCounter;
      this.metrics.skippedFrames = this.skippedFrames;
      this.fpsCounter = 0;
      this.modelFpsCounter = 0;
      this.skippedFrames = 0;
      this.fpsTimestamp = timestamp;
    }
  }

  private logMetrics(): void {
    console.log(
      `[Segmentation] FPS: ${this.metrics.fps} | ` +
      `Model: ${this.metrics.modelInferenceMs.toFixed(1)}ms @ ${this.metrics.modelFps}fps | ` +
      `Pipeline: ${this.metrics.pipelineMs.toFixed(1)}ms | ` +
      `Total: ${this.metrics.totalFrameMs.toFixed(1)}ms | ` +
      `Skipped: ${this.metrics.skippedFrames}`,
    );
  }

  private log(msg: string, data?: Record<string, unknown>): void {
    if (this.opts.debug) {
      console.log(`[Segmentation] ${msg}`, data || '');
    }
  }
}
