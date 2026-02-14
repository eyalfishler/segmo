/**
 * WebGL Shader Sources for Segmentation Post-Processing Pipeline
 *
 * Pipeline stages:
 * 1. Temporal smoothing with hysteresis (kills flickering)
 * 2. Edge-aware bilateral upsample (snaps mask to RGB edges)
 * 3. Gaussian edge feathering (soft transitions)
 * 4. Final compositing (foreground + background blend)
 */

// Shared vertex shader — full-screen quad
export const VERTEX_SHADER = `#version 300 es
in vec2 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}`;

/**
 * Stage 1: Temporal Smoothing with Hysteresis + Motion Awareness
 *
 * Uses different blend rates for foreground-appearing vs foreground-disappearing pixels.
 * Faster to add (responsive), slower to remove (stable edges).
 * Also applies a soft threshold to clean up noisy model output.
 *
 * MOTION AWARENESS (per Google Meet's approach):
 * When motion is detected (mask changed significantly between frames),
 * temporal smoothing is reduced so the mask tracks fast movement.
 * When still, smoothing is increased for maximum stability.
 */
export const TEMPORAL_SMOOTH_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_currentMask;   // Raw model output this frame
uniform sampler2D u_previousMask;  // Smoothed mask from last frame
uniform sampler2D u_motionMap;     // Per-pixel motion magnitude (0=still, 1=fast)
uniform float u_appearRate;        // Blend rate when pixel becomes foreground (0.6-0.85)
uniform float u_disappearRate;     // Blend rate when pixel becomes background (0.3-0.5)
uniform float u_threshold;         // Soft threshold center (0.5)
uniform float u_softness;          // Soft threshold width (0.1-0.3)
uniform float u_hasMotionMap;      // 1.0 if motion map is available, 0.0 otherwise

void main() {
  float current = texture(u_currentMask, v_texCoord).r;
  float previous = texture(u_previousMask, v_texCoord).r;

  // Soft threshold: maps raw model output to cleaner 0-1 range
  float lo = u_threshold - u_softness;
  float hi = u_threshold + u_softness;
  current = smoothstep(lo, hi, current);

  // Base hysteresis rates
  float appearRate = u_appearRate;
  float disappearRate = u_disappearRate;

  // Motion-aware adjustment (branchless):
  // Lower threshold + tighter range = triggers earlier on smaller movements
  float motion = texture(u_motionMap, v_texCoord).r;
  float motionFactor = smoothstep(0.03, 0.2, motion) * u_hasMotionMap;
  appearRate = mix(appearRate, 0.98, motionFactor);
  // Near-instant fade for vacated pixels — kills trailing
  disappearRate = mix(disappearRate, 0.95, motionFactor);

  // Hysteresis: different rates for appearing vs disappearing (branchless)
  float alpha = mix(disappearRate, appearRate, step(previous, current));
  float smoothed = mix(previous, current, alpha);

  outColor = vec4(smoothed, smoothed, smoothed, 1.0);
}`;

/**
 * Stage 1.5: Morphological Operations (Dilate/Erode)
 *
 * Applies small dilation then erosion ("closing") to the mask.
 * This fills small holes in the foreground and smooths jagged edges.
 * Google Meet uses this before the bilateral filter.
 *
 * The direction of operation depends on the halo problem:
 * - Erode slightly to remove background halos bleeding into foreground
 * - Dilate slightly to prevent foreground pixels being classified as background
 */
export const MORPHOLOGY_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_mask;
uniform vec2 u_texelSize;
uniform float u_operation;    // 0.0 = dilate, 1.0 = erode
uniform float u_radius;       // Kernel radius in texels (0.5-2.0)

void main() {
  float center = texture(u_mask, v_texCoord).r;

  // 3x3 kernel with hoisted multiply
  vec2 step = u_texelSize * u_radius;
  float result = center;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      float s = texture(u_mask, v_texCoord + vec2(float(x), float(y)) * step).r;
      result = mix(max(result, s), min(result, s), u_operation);
    }
  }

  outColor = vec4(result, result, result, 1.0);
}`;

/**
 * Stage 2: Joint Bilateral Upsample
 *
 * When the model runs at lower resolution (e.g., 256x144), this upsamples
 * the mask using the full-resolution RGB frame as an edge guide.
 * Mask edges snap to actual object boundaries in the camera feed.
 */
export const BILATERAL_UPSAMPLE_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_mask;          // Low-res smoothed mask
uniform sampler2D u_guide;         // Full-res camera frame (RGB guide)
uniform vec2 u_maskSize;           // Mask texture dimensions
uniform vec2 u_guideSize;          // Guide texture dimensions
uniform float u_spatialSigma;      // Spatial kernel sigma (pixels)
uniform float u_rangeSigma;        // Color similarity sigma (0.05-0.15)

// Precomputed spatial distances for 5x5 kernel (avoids int math in loop)
const float sDist[25] = float[25](
  8.0,5.0,4.0,5.0,8.0, 5.0,2.0,1.0,2.0,5.0, 4.0,1.0,0.0,1.0,4.0, 5.0,2.0,1.0,2.0,5.0, 8.0,5.0,4.0,5.0,8.0
);
const vec2 kOff[25] = vec2[25](
  vec2(-2,-2),vec2(-1,-2),vec2(0,-2),vec2(1,-2),vec2(2,-2),
  vec2(-2,-1),vec2(-1,-1),vec2(0,-1),vec2(1,-1),vec2(2,-1),
  vec2(-2, 0),vec2(-1, 0),vec2(0, 0),vec2(1, 0),vec2(2, 0),
  vec2(-2, 1),vec2(-1, 1),vec2(0, 1),vec2(1, 1),vec2(2, 1),
  vec2(-2, 2),vec2(-1, 2),vec2(0, 2),vec2(1, 2),vec2(2, 2)
);

void main() {
  vec3 centerColor = texture(u_guide, v_texCoord).rgb;
  vec2 maskTexelSize = 1.0 / u_maskSize;

  // Precompute reciprocals (avoid division in loop)
  float spatialRecip = 1.0 / (2.0 * u_spatialSigma * u_spatialSigma);
  float rangeRecip = 1.0 / (2.0 * u_rangeSigma * u_rangeSigma);

  // Precompute center luminance and chroma for perceptual distance
  const vec3 lumW = vec3(0.299, 0.587, 0.114);
  float centerLum = dot(centerColor, lumW);

  float totalWeight = 0.0;
  float totalMask = 0.0;

  for (int i = 0; i < 25; i++) {
    vec2 sampleCoord = v_texCoord + kOff[i] * maskTexelSize;

    float spatialWeight = exp(-sDist[i] * spatialRecip);

    // Perceptual distance: separate luminance and chroma components
    // White bg vs light skin have similar luminance but very different chroma
    vec3 sampleColor = texture(u_guide, sampleCoord).rgb;
    vec3 colorDiff = centerColor - sampleColor;
    float lumDiff = dot(colorDiff, lumW);
    vec3 chromaDiff = colorDiff - lumDiff; // pure chroma difference
    // Weight chroma 3x: amplifies skin-vs-white distinction
    float dist2 = lumDiff * lumDiff + dot(chromaDiff, chromaDiff) * 3.0;
    float rangeWeight = exp(-dist2 * rangeRecip);

    float weight = spatialWeight * rangeWeight;
    totalMask += texture(u_mask, sampleCoord).r * weight;
    totalWeight += weight;
  }

  float result = mix(texture(u_mask, v_texCoord).r, totalMask / max(totalWeight, 0.0001), step(0.0001, totalWeight));
  outColor = vec4(result, result, result, 1.0);
}`;

/**
 * Stage 3: Edge Feathering
 *
 * Applies Gaussian blur ONLY to edge pixels (where mask is between 0.05-0.95).
 * Interior foreground/background pixels are left sharp.
 * This creates the soft, natural-looking edges Google Meet is known for.
 */
export const EDGE_FEATHER_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_mask;
uniform vec2 u_texelSize;          // 1.0 / texture dimensions
uniform float u_featherRadius;     // Blur radius in texels (2.0-5.0)
uniform float u_edgeLow;           // Edge detection low threshold (0.05)
uniform float u_edgeHigh;          // Edge detection high threshold (0.95)

void main() {
  float center = texture(u_mask, v_texCoord).r;

  // Detect edge: check 8 neighbors using precomputed offsets
  vec2 edgeStep = u_texelSize * 2.0;
  float maxDiff = 0.0;
  maxDiff = max(maxDiff, abs(center - texture(u_mask, v_texCoord + vec2(-edgeStep.x, -edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(u_mask, v_texCoord + vec2(0.0, -edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(u_mask, v_texCoord + vec2(edgeStep.x, -edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(u_mask, v_texCoord + vec2(-edgeStep.x, 0.0)).r));
  maxDiff = max(maxDiff, abs(center - texture(u_mask, v_texCoord + vec2(edgeStep.x, 0.0)).r));
  maxDiff = max(maxDiff, abs(center - texture(u_mask, v_texCoord + vec2(-edgeStep.x, edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(u_mask, v_texCoord + vec2(0.0, edgeStep.y)).r));
  maxDiff = max(maxDiff, abs(center - texture(u_mask, v_texCoord + vec2(edgeStep.x, edgeStep.y)).r));

  float edgeness = smoothstep(0.02, 0.15, maxDiff);

  if (edgeness < 0.01) {
    outColor = vec4(center, center, center, 1.0);
    return;
  }

  // 5x5 Gaussian blur with precomputed reciprocal and distance table
  const float bDist[25] = float[25](
    8.0,5.0,4.0,5.0,8.0, 5.0,2.0,1.0,2.0,5.0, 4.0,1.0,0.0,1.0,4.0, 5.0,2.0,1.0,2.0,5.0, 8.0,5.0,4.0,5.0,8.0
  );
  const vec2 bOff[25] = vec2[25](
    vec2(-2,-2),vec2(-1,-2),vec2(0,-2),vec2(1,-2),vec2(2,-2),
    vec2(-2,-1),vec2(-1,-1),vec2(0,-1),vec2(1,-1),vec2(2,-1),
    vec2(-2, 0),vec2(-1, 0),vec2(0, 0),vec2(1, 0),vec2(2, 0),
    vec2(-2, 1),vec2(-1, 1),vec2(0, 1),vec2(1, 1),vec2(2, 1),
    vec2(-2, 2),vec2(-1, 2),vec2(0, 2),vec2(1, 2),vec2(2, 2)
  );

  float radiusSq = u_featherRadius * u_featherRadius;
  float sigmaRecip = 0.5 / radiusSq; // = 1/(2*sigma^2) where sigma=featherRadius
  vec2 blurStep = u_texelSize * u_featherRadius;
  float blurred = 0.0;
  float totalWeight = 0.0;

  for (int i = 0; i < 25; i++) {
    float weight = exp(-bDist[i] * radiusSq * sigmaRecip);
    blurred += texture(u_mask, v_texCoord + bOff[i] * blurStep).r * weight;
    totalWeight += weight;
  }

  blurred /= totalWeight;

  // Blend between sharp and blurred based on edgeness
  float result = mix(center, blurred, edgeness);
  outColor = vec4(result, result, result, 1.0);
}`;

/**
 * Stage 4: Final Compositing
 *
 * Blends foreground camera frame with the background (blur, image, or color).
 * Supports multiple background modes.
 */
export const COMPOSITE_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_camera;        // Full-res camera frame
uniform sampler2D u_mask;          // Final processed mask
uniform sampler2D u_background;    // Background texture (blurred frame, image, etc.)
uniform int u_backgroundMode;      // 0=blur, 1=image, 2=color
uniform vec3 u_backgroundColor;    // Solid color background
uniform vec2 u_texelSize;          // 1.0 / frame dimensions
uniform vec2 u_cropOffset;         // Auto-frame crop offset (0,0 when no crop)
uniform vec2 u_cropSize;           // Auto-frame crop size (1,1 when no crop)

// Cross-shaped sample pattern: wider reach for fg/bg color estimation (13 samples)
const vec2 mOff[13] = vec2[13](
  vec2(0, 0),
  vec2(-1,0),vec2(1,0),vec2(0,-1),vec2(0,1),          // 1px cross
  vec2(-2,0),vec2(2,0),vec2(0,-2),vec2(0,2),          // 2px cross
  vec2(-3,0),vec2(3,0),vec2(0,-3),vec2(0,3)           // 3px cross
);

void main() {
  float rawMask = texture(u_mask, v_texCoord).r;
  vec3 I = texture(u_camera, v_texCoord).rgb;

  // Edge-adaptive sharpening using camera gradient
  vec3 dx = I - texture(u_camera, v_texCoord + vec2(u_texelSize.x, 0.0)).rgb;
  vec3 dy = I - texture(u_camera, v_texCoord + vec2(0.0, u_texelSize.y)).rgb;
  float edgeStrength = dot(dx, dx) + dot(dy, dy);
  float sharpness = smoothstep(0.001, 0.02, edgeStrength);
  float lo = mix(0.15, 0.35, sharpness);
  float hi = mix(0.85, 0.65, sharpness);
  float mask = smoothstep(lo, hi, rawMask);

  // Background UV: reverse auto-frame crop so background stays fixed on screen
  vec2 bgUV = (v_texCoord - u_cropOffset) / u_cropSize;

  // New background color
  vec4 bgTex = texture(u_background, bgUV);
  float isColor = step(1.5, float(u_backgroundMode));
  vec3 newBg = mix(bgTex.rgb, u_backgroundColor, isColor);

  // Default output: standard alpha composite
  vec3 result = mix(newBg, I, mask);

  // Foreground recovery in transition zone:
  // Camera pixels here are contaminated: I = F_true * alpha + B_old * (1-alpha)
  // We want: output = F_true * alpha + B_new * (1-alpha)
  // Therefore: output = I + (B_new - B_old) * (1 - alpha)
  // This mathematically removes the old background's color contribution.
  // Foreground recovery: wider zone [0.02, 0.98] to catch misclassified edge pixels
  float inTransition = step(0.02, mask) * step(mask, 0.98);
  if (inTransition > 0.5) {
    vec3 fgColor = vec3(0.0);
    vec3 bgColor = vec3(0.0);
    float fgWeight = 0.0;
    float bgWeight = 0.0;
    vec2 sampleStep = u_texelSize * 4.0;

    for (int i = 0; i < 13; i++) {
      vec2 sc = v_texCoord + mOff[i] * sampleStep;
      float m = texture(u_mask, sc).r;
      vec3 col = texture(u_camera, sc).rgb;
      float dist = length(mOff[i]);
      float proximity = 1.0 / (1.0 + dist);
      float fw = smoothstep(0.6, 0.9, m) * proximity;
      float bw = smoothstep(0.4, 0.1, m) * proximity;
      fgColor += col * fw;
      fgWeight += fw;
      bgColor += col * bw;
      bgWeight += bw;
    }

    float hasBoth = step(0.01, fgWeight) * step(0.01, bgWeight);
    if (hasBoth > 0.5) {
      vec3 F = fgColor / fgWeight;
      vec3 B = bgColor / bgWeight;
      vec3 FB = F - B;
      float denom = dot(FB, FB);

      // Chroma-aware color separation gate
      const vec3 lumW2 = vec3(0.299, 0.587, 0.114);
      float fbLumDiff = dot(FB, lumW2);
      vec3 fbChromaDiff = FB - fbLumDiff;
      float perceptualDenom = fbLumDiff * fbLumDiff + dot(fbChromaDiff, fbChromaDiff) * 3.0;
      float colorSeparation = smoothstep(0.02, 0.08, perceptualDenom);

      // Matted alpha from closed-form equation
      float mattedAlpha = clamp(dot(I - B, FB) / max(denom, 0.01), 0.0, 1.0);

      // Blend factor: stays strong at high mask values to catch misclassified pixels.
      // Old: (1-abs(rawMask*2-1)) peaked at 0.5, was only 0.1 at rawMask=0.95
      // New: flat 1.0 for rawMask 0.15-0.9, gentle fade at extremes
      float blendFactor = smoothstep(0.02, 0.15, rawMask) * (1.0 - smoothstep(0.9, 1.0, rawMask)) * colorSeparation;
      float alpha = mix(mask, mattedAlpha, blendFactor * 0.8);

      // Foreground recovery: subtract old bg contribution, add new bg
      vec3 recovered = I + (newBg - B) * (1.0 - alpha);
      result = mix(result, clamp(recovered, 0.0, 1.0), blendFactor);
    }
  }

  outColor = vec4(result, 1.0);
}`;

/**
 * Background Blur Shader (two-pass separable Gaussian)
 *
 * Run horizontally then vertically for efficient large-radius blur.
 * Uses 13-tap kernel for strong background blur effect.
 */
export const BLUR_PASS_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_source;
uniform vec2 u_direction;          // (1/w, 0) for horizontal, (0, 1/h) for vertical
uniform float u_radius;            // Blur radius multiplier

// 13-tap Gaussian weights (pre-computed, sigma ≈ 4)
const float weights[7] = float[7](
  0.159577, 0.147308, 0.115877, 0.077674, 0.044368, 0.021596, 0.008958
);

void main() {
  vec4 color = texture(u_source, v_texCoord) * weights[0];

  for (int i = 1; i <= 6; i++) {
    vec2 offset = u_direction * float(i) * u_radius;
    color += texture(u_source, v_texCoord + offset) * weights[i];
    color += texture(u_source, v_texCoord - offset) * weights[i];
  }

  outColor = color;
}`;

/**
 * Light Wrapping Shader (optional, advanced)
 *
 * Adds subtle light spill from background onto foreground edges
 * to prevent the "cut out and pasted on" look. Google Meet uses this.
 */
export const LIGHT_WRAP_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_composite;     // Already composited frame
uniform sampler2D u_background;    // Blurred background
uniform sampler2D u_mask;          // Edge mask
uniform float u_wrapStrength;      // Light wrap intensity (0.05-0.15)

void main() {
  vec4 comp = texture(u_composite, v_texCoord);
  vec4 bg = texture(u_background, v_texCoord);
  float mask = texture(u_mask, v_texCoord).r;

  // Light wrap only on narrow edge band
  float edgeMask = smoothstep(0.25, 0.45, mask) * (1.0 - smoothstep(0.55, 0.75, mask));

  // Blend background light into edge pixels
  vec4 wrapped = mix(comp, bg, edgeMask * u_wrapStrength);

  outColor = wrapped;
}`;

/**
 * Color Temperature Matching Shader
 *
 * When using background replacement (not blur), matches the replacement image's
 * color temperature and brightness to the foreground. This prevents the common
 * problem where a warm-lit person looks "pasted onto" a cool-toned background.
 *
 * Computes per-channel gain/bias from foreground edge pixels and applies
 * a subtle correction to the background image.
 */
export const COLOR_MATCH_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_background;       // Original background replacement image
uniform sampler2D u_camera;           // Camera frame (for color reference)
uniform sampler2D u_mask;             // Segmentation mask
uniform vec3 u_fgMeanColor;           // Mean color of foreground edge band (computed on CPU)
uniform vec3 u_bgMeanColor;           // Mean color of background image (computed on CPU)
uniform float u_matchStrength;        // How much to match (0.0-0.5, subtle)

void main() {
  vec4 bg = texture(u_background, v_texCoord);

  // Compute per-channel color correction
  // Simple gain: adjust background channels toward foreground's color temperature
  vec3 correction = u_fgMeanColor / max(u_bgMeanColor, vec3(0.01));

  // Clamp correction to reasonable range (don't over-correct)
  correction = clamp(correction, vec3(0.7), vec3(1.4));

  // Apply subtle correction
  vec3 matched = mix(bg.rgb, bg.rgb * correction, u_matchStrength);

  outColor = vec4(matched, bg.a);
}`;

/**
 * Mask Shift Shader (motion compensation)
 *
 * Shifts mask UV coordinates to predict person's current position
 * on interpolated frames. The shift is computed from centroid velocity
 * tracked across model frames.
 */
export const MASK_SHIFT_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_mask;
uniform vec2 u_shift;

void main() {
  vec2 shifted = clamp(v_texCoord + u_shift, 0.0, 1.0);
  float m = texture(u_mask, shifted).r;
  outColor = vec4(m, m, m, 1.0);
}`;

/**
 * Crop/Zoom Shader (for auto-framing)
 *
 * Simple passthrough that samples from a sub-region of the input texture.
 * Replaces the expensive CPU canvas copy with a single GPU draw call.
 */
export const CROP_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_source;
uniform vec2 u_cropOffset;    // (x, y) of crop region in 0-1
uniform vec2 u_cropSize;      // (width, height) of crop region in 0-1

void main() {
  vec2 uv = u_cropOffset + v_texCoord * u_cropSize;
  outColor = texture(u_source, uv);
}`;
