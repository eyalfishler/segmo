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

// === Diagnostics types ===

export type DiagnosticsLevel = 'off' | 'summary';

export interface DiagnosticEvent {
  timestamp: number;
  clientId: string | null;
  type: 'summary' | 'init';
  data: DiagnosticSummary | DiagnosticInit;
}

export interface DiagnosticInit {
  resolution: { width: number; height: number };
  modelResolution: { width: number; height: number };
  useWorker: boolean;
  modelDelegate: string;
  quality: string;
  autoFrame: boolean;
  backgroundMode: string;
  userAgent: string;
  platform: string;
  hardwareConcurrency: number;
  deviceMemory: number | null;
  devicePixelRatio: number;
  screenResolution: { width: number; height: number };
  gpu: string;
  gpuVendor: string;
  maxTextureSize: number;
  maxRenderbufferSize: number;
  webglVersion: string;
  shadingLanguageVersion: string;
  connectionType: string | null;
  connectionEffectiveType: string | null;
  connectionDownlink: number | null;
}

export interface DiagnosticSummary {
  fps: number;
  modelFps: number;
  avgModelMs: number;
  avgPipelineMs: number;
  avgTotalMs: number;
  p95TotalMs: number;
  droppedFrames: number;
  qualityTier: number;
  qualityLabel: string;
  roiCrop: { x: number; y: number; w: number; h: number } | null;
  autoFrameZoom: number;
  maskCoverage: number;
  bboxAtEdgeCount: number;
  maskEmptyCount: number;
  webglContextLost: boolean;
  /** Base64 JPEG of processed output (only when diagnosticsIncludeImage is true) */
  image: string | null;
  /** Debug log messages accumulated during this interval */
  logs: string[];
}

export interface DiagnosticSnapshot {
  clientId: string | null;
  init: DiagnosticInit | null;
  metrics: PerformanceMetrics;
  roiCrop: { x: number; y: number; w: number; h: number } | null;
  autoFrameCrop: { x: number; y: number; w: number; h: number; zoom: number };
  qualityTier: number;
  qualityLabel: string;
  maskCoverage: number;
  motionVector: { vx: [number, number, number]; vy: number };
  bboxAtEdgeCount: number;
  uptime: number;
  /** Base64 JPEG of processed output (only when diagnosticsIncludeImage is true) */
  image: string | null;
  /** Debug log messages */
  logs: string[];
}

/** Result of SegmentationProcessor.checkCapabilities() */
export interface CapabilityCheckResult {
  /** True if all hard requirements are met — segmo will work */
  supported: boolean;
  /** Human-readable reasons why segmo is not supported (empty if supported) */
  unsupportedReasons: string[];
  /** Individual capability checks */
  capabilities: {
    offscreenCanvas: boolean;
    webgl2: boolean;
    extColorBufferFloat: boolean;
    mediaStreamTrackProcessor: boolean;
    mediaStreamTrackGenerator: boolean;
    videoFrame: boolean;
    transformStream: boolean;
    oesTextureFloatLinear: boolean;
    webWorkers: boolean;
    createImageBitmap: boolean;
  };
  /** Warnings for soft requirements that are missing (non-blocking) */
  warnings: string[];
}

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
  quality?: 'low' | 'medium' | 'high' | 'ultra';
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
  /** Keep background fixed in screen space during auto-frame (default: false) */
  backgroundFixed?: boolean;
  /** Diagnostics callback — receives periodic summary and init events */
  onDiagnostic?: (event: DiagnosticEvent) => void;
  /** Diagnostics level: 'off' or 'summary' (default: 'off') */
  diagnosticsLevel?: DiagnosticsLevel;
  /** Interval in ms between summary diagnostic events (default: 5000) */
  diagnosticsIntervalMs?: number;
  /** Include a low-res JPEG screenshot in diagnostic events (default: false) */
  diagnosticsIncludeImage?: boolean;
  /** Client identifier included in every diagnostic event (default: null) */
  clientId?: string | null;
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
    appearRate: 0.8,
    disappearRate: 0.4,
    featherRadius: 2.5,
    rangeSigma: 0.12,
    lightWrap: true,
    morphology: true,
    blurRadius: 10,
    modelWidth: 256,
    modelHeight: 144,
    modelFps: 12,
  },
  high: {
    appearRate: 0.75,
    disappearRate: 0.35,
    featherRadius: 3.0,
    rangeSigma: 0.1,
    lightWrap: true,
    morphology: true,
    blurRadius: 12,
    modelWidth: 256,
    modelHeight: 144,
    modelFps: 24,
  },
  ultra: {
    appearRate: 0.7,
    disappearRate: 0.35,
    featherRadius: 1.5,
    rangeSigma: 0.08,
    lightWrap: true,
    morphology: true,
    blurRadius: 12,
    modelWidth: 256,
    modelHeight: 144,
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
  private workerInferenceMs = 0;
  private workerHasFreshMask = false;

  // Mask motion compensation: 3-zone velocity prediction
  private maskVx: [number, number, number] = [0, 0, 0];
  private maskVy = 0;
  private interpFrameCount = 0;

  // Diagnostics
  private diagLevel: DiagnosticsLevel = 'off';
  private diagTimer: ReturnType<typeof setInterval> | null = null;
  private diagInitTime = 0;
  private diagBBoxAtEdgeCount = 0;
  private diagMaskEmptyCount = 0;
  private diagMaskCoverageAccum = 0;
  private diagMaskCoverageCount = 0;
  private diagModelMsAccum = 0;
  private diagModelFrameCount = 0;
  private diagPipelineMsAccum = 0;
  private diagTotalMsAccum = 0;
  private diagTotalMsWindow: number[] = [];
  private diagFrameCount = 0;
  private diagInitData: DiagnosticInit | null = null;
  private diagLastMaskCoverage = 0;
  private diagLastOutput: OffscreenCanvas | null = null;
  private diagCaptureCanvas: HTMLCanvasElement | null = null;
  private diagLogs: string[] = [];

  /**
   * Check browser capabilities required by segmo.
   * Call before constructing a processor to verify the browser supports all required APIs.
   * Synchronous, no side effects (cleans up test resources).
   */
  static checkCapabilities(): CapabilityCheckResult {
    const unsupportedReasons: string[] = [];
    const warnings: string[] = [];

    // Hard: OffscreenCanvas
    const offscreenCanvas = typeof OffscreenCanvas !== 'undefined';
    if (!offscreenCanvas) {
      unsupportedReasons.push('OffscreenCanvas is not available — required for WebGL2 rendering pipeline.');
    }

    // Hard: WebGL2 + extensions (create a test context and clean up)
    let webgl2 = false;
    let extColorBufferFloat = false;
    let oesTextureFloatLinear = false;

    if (offscreenCanvas) {
      try {
        const testCanvas = new OffscreenCanvas(1, 1);
        const gl = testCanvas.getContext('webgl2');
        if (gl) {
          webgl2 = true;
          extColorBufferFloat = gl.getExtension('EXT_color_buffer_float') !== null;
          oesTextureFloatLinear = gl.getExtension('OES_texture_float_linear') !== null;
          gl.getExtension('WEBGL_lose_context')?.loseContext();
        }
      } catch { /* WebGL2 creation threw */ }
    }

    if (!webgl2) {
      unsupportedReasons.push('WebGL2 is not available — required for GPU post-processing pipeline.');
    }
    if (webgl2 && !extColorBufferFloat) {
      unsupportedReasons.push('WebGL2 extension EXT_color_buffer_float is not available — required for RGBA16F framebuffer rendering.');
    }

    // Soft: Insertable Streams / WebCodecs (enables zero-copy path; canvas fallback used otherwise)
    const mediaStreamTrackProcessor = typeof MediaStreamTrackProcessor !== 'undefined';
    const mediaStreamTrackGenerator = typeof MediaStreamTrackGenerator !== 'undefined';
    const videoFrame = typeof VideoFrame !== 'undefined';
    const transformStream = typeof TransformStream !== 'undefined';

    if (!mediaStreamTrackProcessor || !mediaStreamTrackGenerator) {
      warnings.push('Insertable Streams not available — using canvas captureStream fallback (slightly higher latency).');
    }

    // Soft: OES_texture_float_linear
    if (webgl2 && !oesTextureFloatLinear) {
      warnings.push('OES_texture_float_linear missing — mask edges may appear blockier.');
    }

    // Soft: Web Workers
    const webWorkers = typeof Worker !== 'undefined';
    if (!webWorkers) {
      warnings.push('Web Workers not available — model inference will run on the main thread.');
    }

    // Soft: createImageBitmap
    const hasCreateImageBitmap = typeof createImageBitmap !== 'undefined';
    if (!hasCreateImageBitmap) {
      warnings.push('createImageBitmap not available — worker-based inference (useWorker: true) will not work.');
    }

    return {
      supported: unsupportedReasons.length === 0,
      unsupportedReasons,
      capabilities: {
        offscreenCanvas,
        webgl2,
        extColorBufferFloat,
        mediaStreamTrackProcessor,
        mediaStreamTrackGenerator,
        videoFrame,
        transformStream,
        oesTextureFloatLinear,
        webWorkers,
        createImageBitmap: hasCreateImageBitmap,
      },
      warnings,
    };
  }

  constructor(options: SegmentationProcessorOptions = {}) {
    this.opts = {
      backgroundMode: 'blur',
      blurRadius: 12,
      backgroundColor: '#00FF00',
      backgroundImage: null,
      modelFps: 0,
      outputFps: 30,
      modelConfig: {},
      quality: 'medium',
      debug: false,
      adaptive: true,
      adaptiveConfig: {},
      autoFrame: {},
      useWorker: false,
      backgroundFixed: false,
      onDiagnostic: () => { },
      diagnosticsLevel: 'off',
      diagnosticsIntervalMs: 5000,
      diagnosticsIncludeImage: false,
      clientId: null,
      ...options,
    };

    this.qualityPreset = QUALITY_PRESETS[this.opts.quality] ?? QUALITY_PRESETS.high;
    this.modelInterval = 1000 / (this.opts.modelFps || this.qualityPreset.modelFps);
    this.diagLevel = this.opts.diagnosticsLevel;

    // Initialize auto-framer
    this.autoFramer = new AutoFramer(this.opts.autoFrame);

    // Initialize adaptive quality controller
    if (this.opts.adaptive) {
      this.adaptive = new AdaptiveQualityController({
        debug: this.opts.debug,
        ...this.opts.adaptiveConfig,
      });

      this.adaptive.onApply((level: QualityLevel) => {
        this.diagLog(`quality-change: tier=${level.tier} label=${level.label} modelFps=${level.modelFps}`);
        this.modelInterval = 1000 / level.modelFps;
        this.pipeline?.updateOptions({
          appearRate: level.appearRate,
          disappearRate: level.disappearRate,
          featherRadius: level.featherRadius,
          rangeSigma: level.rangeSigma,
          blurRadius: level.blurRadius,
          lightWrap: level.lightWrap,
          morphology: level.morphology,
        });
      });
    }
  }

  /**
   * Initialize the processor. Call once before processing frames.
   * Can be called during LiveKit track processor init.
   */
  async init(width: number, height: number): Promise<void> {
    // Coerce to positive integers — OffscreenCanvas requires unsigned long
    width = Math.max(1, Math.round(width) || 1280);
    height = Math.max(1, Math.round(height) || 720);

    // Pre-flight capability check — fail fast with clear diagnostics
    const caps = SegmentationProcessor.checkCapabilities();
    if (!caps.supported) {
      const reasons = caps.unsupportedReasons.join('\n  - ');
      throw new Error(
        `[segmo] Browser does not meet minimum requirements:\n  - ${reasons}\n\nCall SegmentationProcessor.checkCapabilities() for details.`
      );
    }

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

    // Initialize diagnostics early so this.log() captures lifecycle events
    this.diagInitTime = performance.now();

    const preset = this.qualityPreset;

    // Initialize model (skip MediaPipe load when worker handles inference)
    this.model = new SegmentationModel({
      outputWidth: preset.modelWidth,
      outputHeight: preset.modelHeight,
      delegate: 'GPU',
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
      backgroundFixed: this.opts.backgroundFixed,
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
      // Start at ultra (tier 0) — downgrade is fast (2 bad windows ≈ 2s) while
      // upgrade from a lower tier is slow (5 good windows ≈ 5s), so starting
      // high avoids visible quality ramp-up at startup.
      this.adaptive.setTier(0);
      this.adaptive.unlock(); // Allow auto-adjustment after initial setting
    }

    // Initialize Web Worker for off-main-thread inference (optional)
    // Safari's OffscreenCanvas in workers doesn't update drawImage between frames,
    // causing the mask to freeze on the first frame. Use main-thread path instead.
    const isSafari = typeof navigator !== 'undefined' &&
      /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent);
    if (isSafari && this.opts.useWorker) {
      this.log('Safari detected — disabling Web Worker, using main-thread inference');
      await this.model.init();
    }
    if (this.opts.useWorker && !isSafari) {
      try {
        this.workerClient = new ModelWorkerClient({
          outputWidth: preset.modelWidth,
          outputHeight: preset.modelHeight,
          delegate: 'GPU',
          ...this.opts.modelConfig,
        });
        this.workerClient.onMaskReady((result) => {
          this.workerMask = result.mask;
          this.workerMotion = result.motion;
          this.workerBBox = result.bbox;
          this.workerInferenceMs = result.inferenceMs;
          this.workerHasFreshMask = true;
        });
        await this.workerClient.init();
        this.log('Worker initialized — inference runs off main thread');
      } catch (e) {
        this.log('Worker init failed, falling back to main thread', { error: String(e) });
        this.diagLog(`worker-fallback: ${String(e)}`);
        this.workerClient?.destroy();
        this.workerClient = null;
        await this.model.init();
      }
    }

    // Start periodic diagnostics emission
    this.startDiagnostics();
    this.diagLog(`init: ${width}x${height} model=${this.model!.maskWidth}x${this.model!.maskHeight} worker=${this.opts.useWorker} quality=${this.opts.quality}`);
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
        // Update model state from worker results (model.segment() isn't called in worker path)
        this.model.updateBBoxFromExternal(this.workerBBox);
        if (this.workerBBox) {
          this.model.updateCentroidFromExternal(this.workerMask, this.workerBBox);
        }
        // Capture motion vector + reset interpolation counter
        const mv = this.model.getMaskMotionVector();
        this.maskVx = mv.vx;
        this.maskVy = mv.vy;
        this.interpFrameCount = 0;

        // const vxW = mv.vx[0] * 0.6 + mv.vx[1] * 0.3 + mv.vx[2] * 0.1;
        // if (Math.abs(vxW) > 0.0005 || Math.abs(mv.vy) > 0.0005) {
        //   console.log(`[Motion] vx=[${mv.vx[0].toFixed(4)}, ${mv.vx[1].toFixed(4)}, ${mv.vx[2].toFixed(4)}] vy=${mv.vy.toFixed(4)} | model frame`);
        // }

        this.collectDiagFromMask(this.workerMask);
        const pipelineStart = performance.now();
        output = this.pipeline.process(frame, this.workerMask, this.workerMotion);
        this.metrics.pipelineMs = performance.now() - pipelineStart;
        this.metrics.modelInferenceMs = this.workerInferenceMs;
      } else {
        // Interpolate with accumulating motion-compensated shift
        this.skippedFrames++;
        this.interpFrameCount++;
        this.metrics.modelInferenceMs = 0;
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

        this.collectDiagFromMask(mask);
        const pipelineStart = performance.now();
        output = this.pipeline.process(frame, mask, motionMap);
        this.metrics.pipelineMs = performance.now() - pipelineStart;
      } else {
        // Model ran but returned no mask — keep modelInferenceMs (it was a real run)
        this.interpFrameCount++;
        const pipelineStart = performance.now();
        output = this.pipeline.processInterpolated(frame, this.getAccumulatedShift());
        this.metrics.pipelineMs = performance.now() - pipelineStart;
      }
    } else {
      // Not time for model — interpolate with motion-compensated mask
      this.skippedFrames++;
      this.interpFrameCount++;
      this.metrics.modelInferenceMs = 0;
      const pipelineStart = performance.now();
      output = this.pipeline.processInterpolated(frame, this.getAccumulatedShift());
      this.metrics.pipelineMs = performance.now() - pipelineStart;
    }

    this.metrics.totalFrameMs = performance.now() - frameStart;
    this.frameCount++;
    this.collectDiagTiming();

    // Feed frame time to adaptive quality controller
    if (this.adaptive) {
      this.adaptive.reportFrame(this.metrics.totalFrameMs);
    }

    if (this.opts.debug && this.frameCount % 60 === 0) {
      this.logMetrics();
    }

    if (this.opts.diagnosticsIncludeImage) this.diagLastOutput = output;
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

    // Prefer Insertable Streams (zero-copy, lowest latency) when available
    if (typeof MediaStreamTrackProcessor !== 'undefined' &&
        typeof MediaStreamTrackGenerator !== 'undefined') {
      return this.createProcessedTrackInsertable(inputTrack);
    }

    // Fallback: canvas captureStream (works in Safari, Firefox, and older Chrome)
    return this.createProcessedTrackCanvas(inputTrack);
  }

  /** Insertable Streams path — Chrome/Edge only, zero-copy */
  private createProcessedTrackInsertable(inputTrack: MediaStreamTrack): MediaStreamTrack {
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

  /** Canvas captureStream path — cross-browser fallback (Safari, Firefox) */
  private createProcessedTrackCanvas(inputTrack: MediaStreamTrack): MediaStreamTrack {
    const settings = inputTrack.getSettings();
    const w = settings.width || 1280;
    const h = settings.height || 720;

    // Hidden video element to read frames from the input track
    const video = document.createElement('video');
    video.srcObject = new MediaStream([inputTrack]);
    video.muted = true;
    video.playsInline = true;
    video.play();

    // Output canvas for captureStream
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = w;
    outputCanvas.height = h;
    const ctx = outputCanvas.getContext('2d')!;

    const fps = settings.frameRate || 30;
    const outputStream = outputCanvas.captureStream(fps);
    const outputTrack = outputStream.getVideoTracks()[0];

    let running = true;

    const drawFrame = () => {
      if (!running || inputTrack.readyState === 'ended') return;

      if (video.readyState >= video.HAVE_CURRENT_DATA) {
        const timestamp = performance.now();
        const output = this.processFrame(video, timestamp);
        if (output) {
          ctx.drawImage(output, 0, 0);
        } else {
          ctx.drawImage(video, 0, 0, w, h);
        }
      }

      requestAnimationFrame(drawFrame);
    };

    video.addEventListener('playing', () => drawFrame());

    // Clean up when output track is stopped
    const origStop = outputTrack.stop.bind(outputTrack);
    outputTrack.stop = () => {
      running = false;
      video.srcObject = null;
      origStop();
    };

    return outputTrack;
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

  /** Set whether background stays fixed during auto-frame crop */
  setBackgroundFixed(fixed: boolean): void {
    this.opts.backgroundFixed = fixed;
    this.pipeline?.updateOptions({ backgroundFixed: fixed });
  }

  /** Set blur radius */
  setBlurRadius(radius: number): void {
    this.opts.blurRadius = radius;
    this.pipeline?.updateOptions({ blurRadius: radius });
  }

  /** Change quality preset */
  setQuality(quality: 'low' | 'medium' | 'high' | 'ultra'): void {
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
    if (this.diagTimer) {
      clearInterval(this.diagTimer);
      this.diagTimer = null;
    }
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

  // === Diagnostics ===

  /** Change diagnostics level at runtime */
  setDiagnosticsLevel(level: DiagnosticsLevel): void {
    this.diagLevel = level;
    if (this.initialized) {
      this.startDiagnostics();
    }
  }

  /** Get an on-demand diagnostic snapshot (works regardless of diagnosticsLevel) */
  exportDiagnosticSnapshot(): DiagnosticSnapshot {
    const crop = this.autoFramer.getCurrentCrop();
    const level = this.adaptive?.getCurrentLevel();
    return {
      clientId: this.opts.clientId ?? null,
      init: this.diagInitData,
      metrics: { ...this.metrics },
      roiCrop: this.personCropRegion ? { ...this.personCropRegion } : null,
      autoFrameCrop: { x: crop.x, y: crop.y, w: crop.width, h: crop.height, zoom: crop.zoom },
      qualityTier: level?.tier ?? -1,
      qualityLabel: level?.label ?? this.opts.quality ?? 'unknown',
      maskCoverage: this.diagLastMaskCoverage,
      motionVector: { vx: [...this.maskVx] as [number, number, number], vy: this.maskVy },
      bboxAtEdgeCount: this.diagBBoxAtEdgeCount,
      uptime: performance.now() - this.diagInitTime,
      image: this.captureDiagImage(),
      logs: [...this.diagLogs],
    };
  }

  // === Private helpers ===

  /** Start or restart diagnostics timer based on current level */
  private startDiagnostics(): void {
    // Clear existing timer
    if (this.diagTimer) {
      clearInterval(this.diagTimer);
      this.diagTimer = null;
    }

    // Capture init data (once)
    if (!this.diagInitData && this.pipeline && this.model) {
      const glInfo = this.pipeline.getWebGLInfo();
      const nav = typeof navigator !== 'undefined' ? navigator : null;
      const scr = typeof screen !== 'undefined' ? screen : null;
      const conn = nav && 'connection' in nav ? (nav as any).connection : null;
      this.diagInitData = {
        resolution: { width: this.width, height: this.height },
        modelResolution: { width: this.model.maskWidth, height: this.model.maskHeight },
        useWorker: this.opts.useWorker,
        modelDelegate: this.workerClient
          ? this.workerClient.actualDelegate
          : this.model.actualDelegate,
        quality: this.opts.quality,
        autoFrame: !!this.opts.autoFrame?.enabled,
        backgroundMode: this.opts.backgroundMode,
        userAgent: nav?.userAgent ?? 'unknown',
        platform: (nav as any)?.userAgentData?.platform ?? nav?.platform ?? 'unknown',
        hardwareConcurrency: nav?.hardwareConcurrency ?? 0,
        deviceMemory: (nav as any)?.deviceMemory ?? null,
        devicePixelRatio: typeof devicePixelRatio !== 'undefined' ? devicePixelRatio : 1,
        screenResolution: { width: scr?.width ?? 0, height: scr?.height ?? 0 },
        gpu: glInfo.renderer,
        gpuVendor: glInfo.vendor,
        maxTextureSize: glInfo.maxTextureSize,
        maxRenderbufferSize: glInfo.maxRenderbufferSize,
        webglVersion: glInfo.version,
        shadingLanguageVersion: glInfo.shadingLanguageVersion,
        connectionType: conn?.type ?? null,
        connectionEffectiveType: conn?.effectiveType ?? null,
        connectionDownlink: conn?.downlink ?? null,
      };
    }

    if (this.diagLevel === 'off') return;

    // Emit init event
    if (this.diagInitData) {
      this.opts.onDiagnostic({
        timestamp: Date.now(),
        clientId: this.opts.clientId ?? null,
        type: 'init',
        data: this.diagInitData,
      });
    }

    // Reset accumulators
    this.resetDiagAccumulators();

    // Start summary timer
    this.diagTimer = setInterval(() => {
      this.emitDiagSummary();
    }, this.opts.diagnosticsIntervalMs);
  }

  /** Emit a diagnostic summary event and reset accumulators */
  private emitDiagSummary(): void {
    // Skip if no frames were processed (e.g. tab was backgrounded)
    if (this.diagFrameCount === 0) return;
    const level = this.adaptive?.getCurrentLevel();
    const crop = this.autoFramer.getCurrentCrop();
    const avgModel = this.diagModelFrameCount > 0 ? this.diagModelMsAccum / this.diagModelFrameCount : 0;
    const avgPipeline = this.diagFrameCount > 0 ? this.diagPipelineMsAccum / this.diagFrameCount : 0;
    const avgTotal = this.diagFrameCount > 0 ? this.diagTotalMsAccum / this.diagFrameCount : 0;
    const avgCoverage = this.diagMaskCoverageCount > 0
      ? this.diagMaskCoverageAccum / this.diagMaskCoverageCount : 0;

    // Compute p95 from frame time window
    let p95 = 0;
    if (this.diagTotalMsWindow.length > 0) {
      const sorted = [...this.diagTotalMsWindow].sort((a, b) => a - b);
      p95 = sorted[Math.floor(sorted.length * 0.95)];
    }

    const summary: DiagnosticSummary = {
      fps: this.metrics.fps,
      modelFps: this.metrics.modelFps,
      avgModelMs: avgModel,
      avgPipelineMs: avgPipeline,
      avgTotalMs: avgTotal,
      p95TotalMs: p95,
      droppedFrames: this.metrics.skippedFrames,
      qualityTier: level?.tier ?? -1,
      qualityLabel: level?.label ?? this.opts.quality ?? 'unknown',
      roiCrop: this.personCropRegion ? { ...this.personCropRegion } : null,
      autoFrameZoom: crop.zoom,
      maskCoverage: avgCoverage,
      bboxAtEdgeCount: this.diagBBoxAtEdgeCount,
      maskEmptyCount: this.diagMaskEmptyCount,
      webglContextLost: this.pipeline?.isContextLost() ?? false,
      image: this.captureDiagImage(),
      logs: [...this.diagLogs],
    };

    this.opts.onDiagnostic({
      timestamp: Date.now(),
      clientId: this.opts.clientId ?? null,
      type: 'summary',
      data: summary,
    });

    this.resetDiagAccumulators();
  }

  /** Reset diagnostics accumulators between summary intervals */
  private resetDiagAccumulators(): void {
    this.diagBBoxAtEdgeCount = 0;
    this.diagMaskEmptyCount = 0;
    this.diagMaskCoverageAccum = 0;
    this.diagMaskCoverageCount = 0;
    this.diagModelMsAccum = 0;
    this.diagModelFrameCount = 0;
    this.diagPipelineMsAccum = 0;
    this.diagTotalMsAccum = 0;
    this.diagTotalMsWindow = [];
    this.diagFrameCount = 0;
    this.diagLogs = [];
  }

  /** Collect diagnostics data from a model frame (mask produced) */
  private collectDiagFromMask(mask: Float32Array | null): void {
    if (this.diagLevel === 'off') return;
    if (!mask) {
      this.diagMaskEmptyCount++;
      return;
    }

    // Mask coverage: count person pixels
    let personPixels = 0;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] > 0.5) personPixels++;
    }
    const coverage = personPixels / mask.length;
    this.diagLastMaskCoverage = coverage;
    this.diagMaskCoverageAccum += coverage;
    this.diagMaskCoverageCount++;

    if (personPixels === 0) {
      this.diagMaskEmptyCount++;
      this.diagLog(`mask-empty: no person detected (0/${mask.length} pixels)`);
    }

    // Bbox-at-edge detection: is the person touching the ROI crop boundary?
    if (this.personCropRegion && this.model) {
      const rawBBox = this.model.getPersonBBox(0); // no padding — raw detection boundary
      if (rawBBox) {
        const crop = this.personCropRegion;
        const edgeThreshold = 0.02;
        const touchesLeft = rawBBox.x < crop.x + edgeThreshold;
        const touchesTop = rawBBox.y < crop.y + edgeThreshold;
        const touchesRight = (rawBBox.x + rawBBox.w) > (crop.x + crop.w - edgeThreshold);
        const touchesBottom = (rawBBox.y + rawBBox.h) > (crop.y + crop.h - edgeThreshold);
        if (touchesLeft || touchesTop || touchesRight || touchesBottom) {
          this.diagBBoxAtEdgeCount++;
        }
      }
    }
  }

  /** Log a debug message to both console and diagnostics stream */
  private diagLog(msg: string): void {
    if (this.diagLevel === 'off') return;
    const entry = `[${(performance.now() - this.diagInitTime).toFixed(0)}ms] ${msg}`;
    console.log(`[segmo] ${msg}`);
    this.diagLogs.push(entry);
    // Cap log buffer to prevent unbounded growth
    if (this.diagLogs.length > 500) this.diagLogs.shift();
  }

  /** Capture a low-res JPEG from the last processed output (sync, ~1ms) */
  private captureDiagImage(): string | null {
    if (!this.opts.diagnosticsIncludeImage || !this.diagLastOutput) return null;
    try {
      // Reuse a small canvas for capture (320px wide, preserving aspect ratio)
      const src = this.diagLastOutput;
      const scale = Math.min(1, 320 / src.width);
      const w = Math.round(src.width * scale);
      const h = Math.round(src.height * scale);
      if (!this.diagCaptureCanvas) {
        this.diagCaptureCanvas = document.createElement('canvas');
      }
      this.diagCaptureCanvas.width = w;
      this.diagCaptureCanvas.height = h;
      const ctx = this.diagCaptureCanvas.getContext('2d')!;
      ctx.drawImage(src, 0, 0, w, h);
      return this.diagCaptureCanvas.toDataURL('image/jpeg', 0.5);
    } catch {
      return null;
    }
  }

  /** Accumulate frame timing for diagnostics */
  private collectDiagTiming(): void {
    if (this.diagLevel === 'off') return;
    if (this.metrics.modelInferenceMs > 0) {
      this.diagModelMsAccum += this.metrics.modelInferenceMs;
      this.diagModelFrameCount++;
    }
    this.diagPipelineMsAccum += this.metrics.pipelineMs;
    this.diagTotalMsAccum += this.metrics.totalFrameMs;
    this.diagTotalMsWindow.push(this.metrics.totalFrameMs);
    this.diagFrameCount++;
  }

  /** 3-zone constant-velocity shift, weighted by zone importance.
   * Head zone 60%, mid 30%, bottom 10%.
   * Capped to ±0.12 — seated person's realistic range. */
  private getAccumulatedShift(): { dx: number; dy: number } {
    const t = this.interpFrameCount;
    const amp = 1.0; // velocities are in full-frame normalized space (~0.01 during movement)
    const vx = this.maskVx[0] * 0.6 + this.maskVx[1] * 0.3 + this.maskVx[2] * 0.1;

    // Dead zone: ignore noise-level velocity to prevent jitter on interpolated frames
    const deadZone = 0.003; // ~4px at 1280px — below this is model noise, not real movement
    if (Math.abs(vx) < deadZone && Math.abs(this.maskVy) < deadZone) {
      return { dx: 0, dy: 0 };
    }

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
    // Always capture in diagnostics (even if debug=false)
    const detail = data ? ` ${JSON.stringify(data)}` : '';
    this.diagLog(`${msg}${detail}`);
  }
}
