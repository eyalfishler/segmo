/**
 * WebGL Post-Processing Pipeline
 *
 * Manages the GPU-side processing chain:
 * Raw mask → Temporal Smooth → Bilateral Upsample → Edge Feather → Composite
 *
 * All intermediate work stays on the GPU (no CPU readback until final output).
 */

import {
  VERTEX_SHADER,
  TEMPORAL_SMOOTH_SHADER,
  MORPHOLOGY_SHADER,
  BILATERAL_UPSAMPLE_SHADER,
  EDGE_FEATHER_SHADER,
  COMPOSITE_SHADER,
  BLUR_PASS_SHADER,
  LIGHT_WRAP_SHADER,
  COLOR_MATCH_SHADER,
  CROP_SHADER,
  MASK_SHIFT_SHADER,
} from './shaders';

export interface PipelineOptions {
  /** Camera frame width */
  width: number;
  /** Camera frame height */
  height: number;
  /** Model output mask width */
  maskWidth: number;
  /** Model output mask height */
  maskHeight: number;
  /** Background mode: 'blur' | 'image' | 'color' */
  backgroundMode: 'blur' | 'image' | 'color';
  /** Background color (hex string, used when mode = 'color') */
  backgroundColor?: string;
  /** Background image URL (used when mode = 'image') */
  backgroundImage?: HTMLImageElement | null;
  /** Blur radius for background blur (default: 12) */
  blurRadius?: number;
  /** Enable light wrapping on edges (default: true) */
  lightWrap?: boolean;
  /** Enable morphological closing on mask (default: true) */
  morphology?: boolean;
  /** Morphology kernel radius (default: 1.0) */
  morphologyRadius?: number;
  /** Color temperature matching strength for image backgrounds (default: 0.2) */
  colorMatchStrength?: number;
  /** Temporal smoothing appear rate (default: 0.75) */
  appearRate?: number;
  /** Temporal smoothing disappear rate (default: 0.35) */
  disappearRate?: number;
  /** Edge feather radius in texels (default: 3.0) */
  featherRadius?: number;
  /** Bilateral upsample range sigma (default: 0.1) */
  rangeSigma?: number;
}

interface ShaderProgram {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation>;
}

interface Framebuffer {
  fbo: WebGLFramebuffer;
  texture: WebGLTexture;
  width: number;
  height: number;
}

export class PostProcessingPipeline {
  private gl: WebGL2RenderingContext;
  private canvas: OffscreenCanvas;

  // Shader programs
  private temporalProg!: ShaderProgram;
  private morphologyProg!: ShaderProgram;
  private bilateralProg!: ShaderProgram;
  private featherProg!: ShaderProgram;
  private compositeProg!: ShaderProgram;
  private blurProg!: ShaderProgram;
  private lightWrapProg!: ShaderProgram;
  private colorMatchProg!: ShaderProgram;
  private cropProg!: ShaderProgram;

  // Framebuffers for ping-pong rendering
  private temporalFBO!: Framebuffer;
  private previousMaskFBO!: Framebuffer;
  private morphologyFBO1!: Framebuffer;
  private morphologyFBO2!: Framebuffer;
  private shiftProg!: ShaderProgram;
  private shiftFBO!: Framebuffer;
  private bilateralFBO!: Framebuffer;
  private featherFBO!: Framebuffer;
  private blurFBO1!: Framebuffer;
  private blurFBO2!: Framebuffer;
  private compositeFBO!: Framebuffer;
  private preCropFBO!: Framebuffer; // For auto-frame: render here then crop to screen

  // Textures
  private cameraTexture!: WebGLTexture;
  private maskTexture!: WebGLTexture;
  private motionTexture!: WebGLTexture;
  private backgroundTexture!: WebGLTexture;

  // Geometry
  private quadVAO!: WebGLVertexArrayObject;

  private opts: Required<PipelineOptions>;
  private isFirstFrame = true;

  // Auto-frame crop (set by processor, applied in final render)
  private cropRect: { x: number; y: number; w: number; h: number } | null = null;

  constructor(options: PipelineOptions) {
    this.opts = {
      backgroundColor: '#00FF00',
      backgroundImage: null,
      blurRadius: 12,
      lightWrap: true,
      morphology: true,
      morphologyRadius: 1.0,
      colorMatchStrength: 0.2,
      appearRate: 0.75,
      disappearRate: 0.35,
      featherRadius: 3.0,
      rangeSigma: 0.1,
      ...options,
    };

    this.canvas = new OffscreenCanvas(this.opts.width, this.opts.height);
    const gl = this.canvas.getContext('webgl2', {
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
      alpha: false,
      antialias: false,
      powerPreference: 'high-performance',
    });

    if (!gl) throw new Error('WebGL2 not supported');
    this.gl = gl;

    // Required for rendering to RGBA16F framebuffer textures
    gl.getExtension('EXT_color_buffer_float');
    // Required for LINEAR filtering on R32F mask/motion textures
    gl.getExtension('OES_texture_float_linear');
    // Flip Y on all texture uploads — video/canvas/image have top-left origin,
    // WebGL has bottom-left. Only affects texImage2D with pixel data, not FBO rendering.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

    this.init();
  }

  private init(): void {
    const gl = this.gl;

    // Compile all shader programs
    this.temporalProg = this.createProgram(VERTEX_SHADER, TEMPORAL_SMOOTH_SHADER, [
      'u_currentMask', 'u_previousMask', 'u_motionMap', 'u_appearRate', 'u_disappearRate',
      'u_threshold', 'u_softness', 'u_hasMotionMap',
    ]);
    this.morphologyProg = this.createProgram(VERTEX_SHADER, MORPHOLOGY_SHADER, [
      'u_mask', 'u_texelSize', 'u_operation', 'u_radius',
    ]);
    this.shiftProg = this.createProgram(VERTEX_SHADER, MASK_SHIFT_SHADER, [
      'u_mask', 'u_shift',
    ]);
    this.bilateralProg = this.createProgram(VERTEX_SHADER, BILATERAL_UPSAMPLE_SHADER, [
      'u_mask', 'u_guide', 'u_maskSize', 'u_guideSize',
      'u_spatialSigma', 'u_rangeSigma',
    ]);
    this.featherProg = this.createProgram(VERTEX_SHADER, EDGE_FEATHER_SHADER, [
      'u_mask', 'u_texelSize', 'u_featherRadius', 'u_edgeLow', 'u_edgeHigh',
    ]);
    this.compositeProg = this.createProgram(VERTEX_SHADER, COMPOSITE_SHADER, [
      'u_camera', 'u_mask', 'u_background', 'u_backgroundMode', 'u_backgroundColor', 'u_texelSize',
    ]);
    this.blurProg = this.createProgram(VERTEX_SHADER, BLUR_PASS_SHADER, [
      'u_source', 'u_direction', 'u_radius',
    ]);
    this.lightWrapProg = this.createProgram(VERTEX_SHADER, LIGHT_WRAP_SHADER, [
      'u_composite', 'u_background', 'u_mask', 'u_wrapStrength',
    ]);
    this.colorMatchProg = this.createProgram(VERTEX_SHADER, COLOR_MATCH_SHADER, [
      'u_background', 'u_camera', 'u_mask', 'u_fgMeanColor',
      'u_bgMeanColor', 'u_matchStrength',
    ]);
    this.cropProg = this.createProgram(VERTEX_SHADER, CROP_SHADER, [
      'u_source', 'u_cropOffset', 'u_cropSize',
    ]);

    // Create framebuffers
    const { width, height, maskWidth, maskHeight } = this.opts;

    // Mask processing at mask resolution first, then upsampled to full res
    this.temporalFBO = this.createFramebuffer(maskWidth, maskHeight);
    this.previousMaskFBO = this.createFramebuffer(maskWidth, maskHeight);
    this.morphologyFBO1 = this.createFramebuffer(maskWidth, maskHeight);
    this.morphologyFBO2 = this.createFramebuffer(maskWidth, maskHeight);
    this.shiftFBO = this.createFramebuffer(maskWidth, maskHeight);

    // Full resolution processing
    this.bilateralFBO = this.createFramebuffer(width, height);
    this.featherFBO = this.createFramebuffer(width, height);
    this.compositeFBO = this.createFramebuffer(width, height);
    this.preCropFBO = this.createFramebuffer(width, height);

    // Blur at half resolution for performance
    const blurW = Math.floor(width / 2);
    const blurH = Math.floor(height / 2);
    this.blurFBO1 = this.createFramebuffer(blurW, blurH);
    this.blurFBO2 = this.createFramebuffer(blurW, blurH);

    // Create input textures
    this.cameraTexture = this.createTexture();
    this.maskTexture = this.createTexture();
    this.motionTexture = this.createTexture();
    this.backgroundTexture = this.createTexture();

    // Upload background image if provided
    if (this.opts.backgroundImage) {
      gl.bindTexture(gl.TEXTURE_2D, this.backgroundTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.opts.backgroundImage);
    }

    // Create fullscreen quad VAO
    this.quadVAO = this.createQuad();

    // Clear previous mask to black (all background)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.previousMaskFBO.fbo);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Process one frame through the entire pipeline.
   *
   * @param cameraFrame - Current camera frame (VideoFrame, ImageBitmap, HTMLVideoElement, etc.)
   * @param maskData - Raw segmentation mask from model (Float32Array or Uint8Array)
   * @param motionMap - Per-pixel motion magnitude from model (optional, Float32Array)
   * @returns The output canvas (can be captured as a VideoFrame)
   */
  process(
    cameraFrame: TexImageSource,
    maskData: Float32Array | Uint8Array,
    motionMap?: Float32Array | null,
  ): OffscreenCanvas {
    const gl = this.gl;
    const { width, height, maskWidth, maskHeight } = this.opts;

    // Upload camera frame to GPU
    gl.bindTexture(gl.TEXTURE_2D, this.cameraTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cameraFrame);

    // Extend mask at frame edges: copy row 2-in from each edge to the outer 2 rows.
    // Prevents boundary artifacts from model low-confidence at truncated body edges,
    // which would otherwise be amplified by bilateral/feathering/erosion kernels.
    if (maskData instanceof Float32Array) {
      this.padMaskEdges(maskData, maskWidth, maskHeight);
    }

    // Upload raw mask to GPU
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    if (maskData instanceof Float32Array) {
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.R32F,
        maskWidth, maskHeight, 0,
        gl.RED, gl.FLOAT, maskData,
      );
    } else {
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.R8,
        maskWidth, maskHeight, 0,
        gl.RED, gl.UNSIGNED_BYTE, maskData,
      );
    }

    // Upload motion map if available
    const hasMotionMap = motionMap != null;
    if (hasMotionMap) {
      gl.bindTexture(gl.TEXTURE_2D, this.motionTexture);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.R32F,
        maskWidth, maskHeight, 0,
        gl.RED, gl.FLOAT, motionMap!,
      );
    }

    // --- Stage 1: Temporal Smoothing (motion-aware) ---
    this.renderToFBO(this.temporalFBO, this.temporalProg, () => {
      this.bindTexture(0, this.maskTexture, 'u_currentMask');
      this.bindTexture(1, this.previousMaskFBO.texture, 'u_previousMask');
      this.bindTexture(2, hasMotionMap ? this.motionTexture : this.maskTexture, 'u_motionMap');
      gl.uniform1f(this.temporalProg.uniforms['u_appearRate'],
        this.isFirstFrame ? 1.0 : this.opts.appearRate);
      gl.uniform1f(this.temporalProg.uniforms['u_disappearRate'],
        this.isFirstFrame ? 1.0 : this.opts.disappearRate);
      gl.uniform1f(this.temporalProg.uniforms['u_threshold'], 0.5);
      gl.uniform1f(this.temporalProg.uniforms['u_softness'], 0.25);
      gl.uniform1f(this.temporalProg.uniforms['u_hasMotionMap'], hasMotionMap ? 1.0 : 0.0);
    });

    // Copy temporal result to previousMask for next frame
    this.copyFramebuffer(this.temporalFBO, this.previousMaskFBO);

    // --- Stage 1.5: Morphological closing (dilate then erode) ---
    // Fills small holes and smooths jagged edges at mask resolution (cheap)
    let maskForUpsample = this.temporalFBO.texture;

    if (this.opts.morphology) {
      const maskTexelSize = [1.0 / maskWidth, 1.0 / maskHeight] as const;

      // Dilate pass (expand foreground slightly)
      this.renderToFBO(this.morphologyFBO1, this.morphologyProg, () => {
        this.bindTexture(0, this.temporalFBO.texture, 'u_mask');
        gl.uniform2f(this.morphologyProg.uniforms['u_texelSize'], ...maskTexelSize);
        gl.uniform1f(this.morphologyProg.uniforms['u_operation'], 0.0); // dilate
        gl.uniform1f(this.morphologyProg.uniforms['u_radius'], this.opts.morphologyRadius);
      });

      // Erode pass (shrink back — net effect: closed small holes, smoother edges)
      this.renderToFBO(this.morphologyFBO2, this.morphologyProg, () => {
        this.bindTexture(0, this.morphologyFBO1.texture, 'u_mask');
        gl.uniform2f(this.morphologyProg.uniforms['u_texelSize'], ...maskTexelSize);
        gl.uniform1f(this.morphologyProg.uniforms['u_operation'], 1.0); // erode
        gl.uniform1f(this.morphologyProg.uniforms['u_radius'], this.opts.morphologyRadius);
      });

      maskForUpsample = this.morphologyFBO2.texture;
    }

    // --- Stage 2: Bilateral Upsample (mask res → full res) ---
    this.renderToFBO(this.bilateralFBO, this.bilateralProg, () => {
      this.bindTexture(0, maskForUpsample, 'u_mask');
      this.bindTexture(1, this.cameraTexture, 'u_guide');
      gl.uniform2f(this.bilateralProg.uniforms['u_maskSize'], maskWidth, maskHeight);
      gl.uniform2f(this.bilateralProg.uniforms['u_guideSize'], width, height);
      gl.uniform1f(this.bilateralProg.uniforms['u_spatialSigma'], 3.0);
      gl.uniform1f(this.bilateralProg.uniforms['u_rangeSigma'], Math.max(this.opts.rangeSigma, 0.15));
    });

    // --- Stage 3: Edge Feathering ---
    this.renderToFBO(this.featherFBO, this.featherProg, () => {
      this.bindTexture(0, this.bilateralFBO.texture, 'u_mask');
      gl.uniform2f(this.featherProg.uniforms['u_texelSize'], 1.0 / width, 1.0 / height);
      gl.uniform1f(this.featherProg.uniforms['u_featherRadius'], this.opts.featherRadius);
      gl.uniform1f(this.featherProg.uniforms['u_edgeLow'], 0.05);
      gl.uniform1f(this.featherProg.uniforms['u_edgeHigh'], 0.95);
    });

    // --- Stage 3.5: Final mask erosion (anti-halo) ---
    // Erode 1px at full resolution to pull mask inward and cut off contaminated edge pixels
    this.renderToFBO(this.bilateralFBO, this.morphologyProg, () => {
      this.bindTexture(0, this.featherFBO.texture, 'u_mask');
      gl.uniform2f(this.morphologyProg.uniforms['u_texelSize'], 1.0 / width, 1.0 / height);
      gl.uniform1f(this.morphologyProg.uniforms['u_operation'], 1.0); // erode
      gl.uniform1f(this.morphologyProg.uniforms['u_radius'], 0.5);
    });

    // --- Generate background ---
    let backgroundTex: WebGLTexture;

    if (this.opts.backgroundMode === 'blur') {
      // Downsample camera to blur FBO and apply two-pass Gaussian
      this.generateBlurredBackground(cameraFrame);
      backgroundTex = this.blurFBO2.texture;
    } else if (this.opts.backgroundMode === 'image' && this.opts.backgroundImage) {
      backgroundTex = this.backgroundTexture;
    } else {
      backgroundTex = this.backgroundTexture; // Will use color uniform
    }

    // --- Stage 4: Compositing (with color decontamination) ---
    const compositeTarget = this.opts.lightWrap ? this.compositeFBO : null;
    const renderComposite = compositeTarget
      ? (setup: () => void) => this.renderToFBO(compositeTarget, this.compositeProg, setup)
      : (setup: () => void) => this.renderToScreen(this.compositeProg, setup);

    renderComposite(() => {
      this.bindTexture(0, this.cameraTexture, 'u_camera');
      this.bindTexture(1, this.bilateralFBO.texture, 'u_mask'); // eroded mask
      this.bindTexture(2, backgroundTex, 'u_background');
      gl.uniform1i(
        this.compositeProg.uniforms['u_backgroundMode'],
        this.opts.backgroundMode === 'blur' ? 0 :
        this.opts.backgroundMode === 'image' ? 1 : 2,
      );
      const [r, g, b] = this.hexToRgb(this.opts.backgroundColor);
      gl.uniform3f(this.compositeProg.uniforms['u_backgroundColor'], r, g, b);
      gl.uniform2f(this.compositeProg.uniforms['u_texelSize'], 1.0 / width, 1.0 / height);
    });

    // --- Stage 5: Light Wrap (subtle BG light spill on edges) ---
    if (this.opts.lightWrap) {
      this.renderToScreen(this.lightWrapProg, () => {
        this.bindTexture(0, this.compositeFBO.texture, 'u_composite');
        this.bindTexture(1, backgroundTex, 'u_background');
        this.bindTexture(2, this.featherFBO.texture, 'u_mask');
        gl.uniform1f(this.lightWrapProg.uniforms['u_wrapStrength'], 0.06);
      });
    }

    this.isFirstFrame = false;
    return this.canvas;
  }

  /**
   * Process using an interpolated mask (for skipped frames).
   * Re-uses the previous smoothed mask without model inference.
   */
  processInterpolated(
    cameraFrame: TexImageSource,
    maskShift?: { dx: number; dy: number },
  ): OffscreenCanvas {
    const gl = this.gl;
    const { width, height } = this.opts;

    // Upload camera frame
    gl.bindTexture(gl.TEXTURE_2D, this.cameraTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cameraFrame);

    // Motion compensation: shift mask to predicted position before bilateral.
    // On interpolated frames, the mask is stale — this shifts it toward where
    // the person has moved based on tracked centroid velocity.
    const hasShift = maskShift && (Math.abs(maskShift.dx) > 0.0001 || Math.abs(maskShift.dy) > 0.0001);
    let maskForBilateral = this.previousMaskFBO.texture;
    if (hasShift) {
      this.renderToFBO(this.shiftFBO, this.shiftProg, () => {
        this.bindTexture(0, this.previousMaskFBO.texture, 'u_mask');
        gl.uniform2f(this.shiftProg.uniforms['u_shift'], maskShift!.dx, maskShift!.dy);
      });
      maskForBilateral = this.shiftFBO.texture;
    }

    // Bilateral upsample with camera guide (snaps shifted mask to current edges)
    this.renderToFBO(this.bilateralFBO, this.bilateralProg, () => {
      this.bindTexture(0, maskForBilateral, 'u_mask');
      this.bindTexture(1, this.cameraTexture, 'u_guide');
      gl.uniform2f(this.bilateralProg.uniforms['u_maskSize'], this.opts.maskWidth, this.opts.maskHeight);
      gl.uniform2f(this.bilateralProg.uniforms['u_guideSize'], width, height);
      gl.uniform1f(this.bilateralProg.uniforms['u_spatialSigma'], 3.0);
      gl.uniform1f(this.bilateralProg.uniforms['u_rangeSigma'], Math.max(this.opts.rangeSigma, 0.15));
    });

    this.renderToFBO(this.featherFBO, this.featherProg, () => {
      this.bindTexture(0, this.bilateralFBO.texture, 'u_mask');
      gl.uniform2f(this.featherProg.uniforms['u_texelSize'], 1.0 / width, 1.0 / height);
      gl.uniform1f(this.featherProg.uniforms['u_featherRadius'], this.opts.featherRadius);
      gl.uniform1f(this.featherProg.uniforms['u_edgeLow'], 0.05);
      gl.uniform1f(this.featherProg.uniforms['u_edgeHigh'], 0.95);
    });

    // Final mask erosion (anti-halo)
    this.renderToFBO(this.bilateralFBO, this.morphologyProg, () => {
      this.bindTexture(0, this.featherFBO.texture, 'u_mask');
      gl.uniform2f(this.morphologyProg.uniforms['u_texelSize'], 1.0 / width, 1.0 / height);
      gl.uniform1f(this.morphologyProg.uniforms['u_operation'], 1.0); // erode
      gl.uniform1f(this.morphologyProg.uniforms['u_radius'], 0.5);
    });

    let backgroundTex: WebGLTexture;
    if (this.opts.backgroundMode === 'blur') {
      this.generateBlurredBackground(cameraFrame);
      backgroundTex = this.blurFBO2.texture;
    } else if (this.opts.backgroundMode === 'image' && this.opts.backgroundImage) {
      backgroundTex = this.backgroundTexture;
    } else {
      backgroundTex = this.backgroundTexture;
    }

    const compositeTarget = this.opts.lightWrap ? this.compositeFBO : null;
    const renderComposite = compositeTarget
      ? (setup: () => void) => this.renderToFBO(compositeTarget, this.compositeProg, setup)
      : (setup: () => void) => this.renderToScreen(this.compositeProg, setup);

    renderComposite(() => {
      this.bindTexture(0, this.cameraTexture, 'u_camera');
      this.bindTexture(1, this.bilateralFBO.texture, 'u_mask'); // eroded mask
      this.bindTexture(2, backgroundTex, 'u_background');
      gl.uniform1i(
        this.compositeProg.uniforms['u_backgroundMode'],
        this.opts.backgroundMode === 'blur' ? 0 :
        this.opts.backgroundMode === 'image' ? 1 : 2,
      );
      const [r, g, b] = this.hexToRgb(this.opts.backgroundColor);
      gl.uniform3f(this.compositeProg.uniforms['u_backgroundColor'], r, g, b);
      gl.uniform2f(this.compositeProg.uniforms['u_texelSize'], 1.0 / width, 1.0 / height);
    });

    if (this.opts.lightWrap) {
      this.renderToScreen(this.lightWrapProg, () => {
        this.bindTexture(0, this.compositeFBO.texture, 'u_composite');
        this.bindTexture(1, backgroundTex, 'u_background');
        this.bindTexture(2, this.featherFBO.texture, 'u_mask');
        gl.uniform1f(this.lightWrapProg.uniforms['u_wrapStrength'], 0.06);
      });
    }

    return this.canvas;
  }

  /** Set auto-frame crop rect. When set, the final render crops/zooms the output on GPU. */
  setCropRect(rect: { x: number; y: number; w: number; h: number } | null): void {
    this.cropRect = rect;
  }

  /** Update pipeline options at runtime */
  updateOptions(opts: Partial<PipelineOptions>): void {
    Object.assign(this.opts, opts);

    if (opts.backgroundImage) {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.backgroundTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, opts.backgroundImage);
    }
  }

  /** Clean up all GPU resources */
  destroy(): void {
    const gl = this.gl;

    // Delete programs
    [this.temporalProg, this.morphologyProg, this.shiftProg, this.bilateralProg, this.featherProg,
     this.compositeProg, this.blurProg, this.lightWrapProg, this.colorMatchProg,
     this.cropProg].forEach(p => {
      gl.deleteProgram(p.program);
    });

    // Delete framebuffers
    [this.temporalFBO, this.previousMaskFBO, this.morphologyFBO1, this.morphologyFBO2, this.shiftFBO,
     this.bilateralFBO, this.featherFBO, this.blurFBO1, this.blurFBO2,
     this.compositeFBO, this.preCropFBO].forEach(fbo => {
      gl.deleteFramebuffer(fbo.fbo);
      gl.deleteTexture(fbo.texture);
    });

    // Delete textures
    gl.deleteTexture(this.cameraTexture);
    gl.deleteTexture(this.maskTexture);
    gl.deleteTexture(this.motionTexture);
    gl.deleteTexture(this.backgroundTexture);

    gl.deleteVertexArray(this.quadVAO);
  }

  // ==========================================================================
  // Private helpers
  // ==========================================================================

  private generateBlurredBackground(cameraFrame: TexImageSource): void {
    const gl = this.gl;
    const blurW = this.blurFBO1.width;
    const blurH = this.blurFBO1.height;

    // Pass 1: Horizontal blur (camera → blurFBO1)
    this.renderToFBO(this.blurFBO1, this.blurProg, () => {
      this.bindTexture(0, this.cameraTexture, 'u_source');
      gl.uniform2f(this.blurProg.uniforms['u_direction'], 1.0 / blurW, 0.0);
      gl.uniform1f(this.blurProg.uniforms['u_radius'], this.opts.blurRadius);
    });

    // Pass 2: Vertical blur (blurFBO1 → blurFBO2)
    this.renderToFBO(this.blurFBO2, this.blurProg, () => {
      this.bindTexture(0, this.blurFBO1.texture, 'u_source');
      gl.uniform2f(this.blurProg.uniforms['u_direction'], 0.0, 1.0 / blurH);
      gl.uniform1f(this.blurProg.uniforms['u_radius'], this.opts.blurRadius);
    });

    // Additional passes for stronger blur
    for (let i = 0; i < 2; i++) {
      this.renderToFBO(this.blurFBO1, this.blurProg, () => {
        this.bindTexture(0, this.blurFBO2.texture, 'u_source');
        gl.uniform2f(this.blurProg.uniforms['u_direction'], 1.0 / blurW, 0.0);
        gl.uniform1f(this.blurProg.uniforms['u_radius'], this.opts.blurRadius * 0.7);
      });

      this.renderToFBO(this.blurFBO2, this.blurProg, () => {
        this.bindTexture(0, this.blurFBO1.texture, 'u_source');
        gl.uniform2f(this.blurProg.uniforms['u_direction'], 0.0, 1.0 / blurH);
        gl.uniform1f(this.blurProg.uniforms['u_radius'], this.opts.blurRadius * 0.7);
      });
    }
  }

  private renderToFBO(fbo: Framebuffer, prog: ShaderProgram, setupUniforms: () => void): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.fbo);
    gl.viewport(0, 0, fbo.width, fbo.height);
    gl.useProgram(prog.program);
    setupUniforms();
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private renderToScreen(prog: ShaderProgram, setupUniforms: () => void): void {
    if (this.cropRect) {
      // Auto-frame active: render to preCropFBO, then crop to screen
      this.renderToFBO(this.preCropFBO, prog, setupUniforms);
      const gl = this.gl;
      const c = this.cropRect;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.opts.width, this.opts.height);
      gl.useProgram(this.cropProg.program);
      this.bindTexture(0, this.preCropFBO.texture, 'u_source');
      gl.uniform2f(this.cropProg.uniforms['u_cropOffset'], c.x, c.y);
      gl.uniform2f(this.cropProg.uniforms['u_cropSize'], c.w, c.h);
      gl.bindVertexArray(this.quadVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    } else {
      // No crop: render directly to screen
      const gl = this.gl;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.opts.width, this.opts.height);
      gl.useProgram(prog.program);
      setupUniforms();
      gl.bindVertexArray(this.quadVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }

  private bindTexture(unit: number, texture: WebGLTexture, uniformName: string): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Find the uniform in the currently active program
    const loc = gl.getUniformLocation(gl.getParameter(gl.CURRENT_PROGRAM), uniformName);
    if (loc) gl.uniform1i(loc, unit);
  }

  private copyFramebuffer(src: Framebuffer, dst: Framebuffer): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, src.fbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst.fbo);
    gl.blitFramebuffer(
      0, 0, src.width, src.height,
      0, 0, dst.width, dst.height,
      gl.COLOR_BUFFER_BIT, gl.NEAREST,
    );
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
  }

  private createProgram(vertSrc: string, fragSrc: string, uniformNames: string[]): ShaderProgram {
    const gl = this.gl;

    const vertShader = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vertShader, vertSrc);
    gl.compileShader(vertShader);
    if (!gl.getShaderParameter(vertShader, gl.COMPILE_STATUS)) {
      throw new Error(`Vertex shader error: ${gl.getShaderInfoLog(vertShader)}`);
    }

    const fragShader = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fragShader, fragSrc);
    gl.compileShader(fragShader);
    if (!gl.getShaderParameter(fragShader, gl.COMPILE_STATUS)) {
      throw new Error(`Fragment shader error: ${gl.getShaderInfoLog(fragShader)}`);
    }

    const program = gl.createProgram()!;
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`);
    }

    gl.deleteShader(vertShader);
    gl.deleteShader(fragShader);

    const uniforms: Record<string, WebGLUniformLocation> = {};
    for (const name of uniformNames) {
      const loc = gl.getUniformLocation(program, name);
      if (loc) uniforms[name] = loc;
    }

    return { program, uniforms };
  }

  private createFramebuffer(width: number, height: number): Framebuffer {
    const gl = this.gl;

    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete: ${status}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, texture, width, height };
  }

  private createTexture(): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private createQuad(): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    // Fullscreen quad (2 triangles as triangle strip)
    const vertices = new Float32Array([
      // position    texCoord
      -1, -1,        0, 0,
       1, -1,        1, 0,
      -1,  1,        0, 1,
       1,  1,        1, 1,
    ]);

    const vbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    // a_position
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    // a_texCoord
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);

    gl.bindVertexArray(null);
    return vao;
  }

  /** Extend mask values at frame edges to prevent boundary artifacts from kernel sampling. */
  private padMaskEdges(mask: Float32Array, w: number, h: number): void {
    const PAD = 4;
    // Bottom PAD rows ← row (h - PAD - 1)
    const srcBot = (h - PAD - 1) * w;
    for (let r = 0; r < PAD; r++) {
      const dst = (h - 1 - r) * w;
      for (let x = 0; x < w; x++) mask[dst + x] = mask[srcBot + x];
    }
    // Top PAD rows ← row PAD
    const srcTop = PAD * w;
    for (let r = 0; r < PAD; r++) {
      const dst = r * w;
      for (let x = 0; x < w; x++) mask[dst + x] = mask[srcTop + x];
    }
    // Left/right PAD cols
    for (let y = 0; y < h; y++) {
      const off = y * w;
      for (let c = 0; c < PAD; c++) {
        mask[off + c] = mask[off + PAD];
        mask[off + w - 1 - c] = mask[off + w - 1 - PAD];
      }
    }
  }

  private hexToRgb(hex: string): [number, number, number] {
    const h = hex.replace('#', '');
    return [
      parseInt(h.substring(0, 2), 16) / 255,
      parseInt(h.substring(2, 4), 16) / 255,
      parseInt(h.substring(4, 6), 16) / 255,
    ];
  }
}
