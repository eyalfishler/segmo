# segmo

> Near Google Meet-quality background segmentation for video conferencing.

Lightweight MediaPipe model + multi-stage GPU post-processing pipeline with alpha matting, temporal smoothing, edge-aware upsampling, and adaptive quality. Drops into LiveKit, WebRTC, or any `MediaStreamTrack`.

## What it does

Takes your webcam feed and produces smooth, artifact-free background blur/replacement at 30fps — approaching the quality Google Meet achieves with temporal stability, sharp edges, hair detail preservation, and no flickering.

## Install

```bash
npm install segmo
```

## Quick Start

### With LiveKit

Drop-in replacement for `@livekit/track-processors`. Implements the official `TrackProcessor` interface via `processedTrack`.

```ts
import { SegmentationProcessor } from 'segmo';
import { Room, createLocalVideoTrack } from 'livekit-client';

// 1. Create processor
const processor = new SegmentationProcessor({
  backgroundMode: 'blur',
  useWorker: true,  // off-main-thread inference
});

// 2. Create video track and attach processor
const videoTrack = await createLocalVideoTrack({
  resolution: { width: 1280, height: 720, frameRate: 30 },
});
await videoTrack.setProcessor(processor.toLiveKitProcessor());

// 3. Connect and publish
const room = new Room();
await room.connect(serverUrl, token);
await room.localParticipant.publishTrack(videoTrack);

// 4. Switch modes at runtime (no re-initialization needed)
processor.setBackgroundMode('blur');
processor.setBlurRadius(14);
processor.setBackgroundMode('color');
processor.setBackgroundColor('#1a1a2e');

// 5. Stop processing
await videoTrack.stopProcessor();

// 6. Clean up
processor.destroy();
await room.disconnect();
```

### Without LiveKit

Works with any `MediaStreamTrack` — WebRTC, recording, or custom video pipelines:

```ts
import { SegmentationProcessor } from 'segmo';

const processor = new SegmentationProcessor({
  backgroundMode: 'blur',
  useWorker: true,
});
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
const outputTrack = await processor.createProcessedTrack(stream.getVideoTracks()[0]);
document.querySelector('video').srcObject = new MediaStream([outputTrack]);
```

## Background Modes

```ts
// Blur (default)
processor.setBackgroundMode('blur');
processor.setBlurRadius(14);

// Custom image
const img = new Image();
img.src = '/backgrounds/office.jpg';
img.onload = () => processor.setBackgroundImage(img);

// Solid color
processor.setBackgroundMode('color');
processor.setBackgroundColor('#1a1a2e');

// Disable
processor.setBackgroundMode('none');
```

All switchable at runtime — no re-initialization.

## Architecture

```
Camera (30fps render loop, matched to model)
  │
  ├──▶ ROI Crop ── person bbox from previous mask → tighter model input (~2x resolution)
  │         │       dead zone + smoothing prevents jitter feedback
  │         ▼
  ├──▶ MediaPipe Selfie Segmenter (CPU WASM+SIMD, 30fps ultra, 256x256)
  │         │
  │         ▼
  │    [Stage 1] Temporal Smoothing ── motion-aware hysteresis (branchless)
  │         │                          0.95 disappear rate for fast trailing fade
  │         ▼
  │    [Stage 1.5] Morphological Closing ── dilate→erode, fills holes
  │         │
  │         ▼
  ├──▶ [Stage 2] Bilateral Upsample ── chroma-weighted perceptual distance (3x chroma)
  │         │                           RGB-guided edge snapping, precomputed tables
  │         ▼
  │    [Stage 3] Edge Feathering ── selective Gaussian blur on edges only
  │         │                       early-exit for non-edge pixels (90%+ skip)
  │         ▼
  │    [Stage 3.5] Mask Erosion ── 0.5px inward pull to cut contaminated edge pixels
  │         │
  │         ▼
  │    [Stage 4] Compositing ──── foreground recovery: output = I + (newBg - B)*(1-α)
  │         │                     alpha matting + adaptive edge sharpening
  │         │                     chroma-aware color separation gate
  │         ▼
  │    [Stage 5] Light Wrap ──── subtle background light spill on edges
  │         │
  │         ▼
  │    [Stage 6] Auto-Frame ──── GPU crop shader for centering/zoom (optional)
  │                               distance-adaptive zoom, smooth tracking
  │
  ▼
Output (30fps, matched to model — zero interpolation at ultra)
```

### Pipeline Stages in Detail

#### ROI Cropping (pre-inference)

**Problem:** The model sees the entire 1280x720 frame scaled to 256x256, so the person occupies maybe 40% of those pixels — edges are blurry at ~3px resolution.

**Solution:** Use the previous frame's segmentation mask to compute a person bounding box, then crop just that region and scale it to fill the full 256x256 input. The person's edges now get ~2x the pixel resolution.

**Stability:** A dead zone prevents jitter feedback — position changes below 3% are ignored (prevents crop→mask→crop oscillation), while size changes above 1.5% are tracked (responds to distance changes). Transitions use 50% exponential smoothing.

#### Stage 1: Temporal Smoothing

**Problem:** The model's raw output flickers frame-to-frame — a pixel might be 0.7 one frame, 0.4 the next, 0.8 after that. This creates distracting shimmer on edges.

**Math:** Per-pixel exponential moving average with asymmetric rates:

```
smoothed = mix(previous, current, alpha)
alpha = current > previous ? appearRate : disappearRate
```

- `appearRate` = 0.7 (foreground appears quickly — responsive tracking)
- `disappearRate` = 0.3 (foreground disappears slowly — stable edges)

**Motion awareness:** When the per-pixel motion map (|current - previous|) detects movement, rates are adjusted via `smoothstep(0.03, 0.2, motion)`:
- Appear rate → 0.98 (near-instant tracking of fast movement)
- Disappear rate → 0.95 (fast fade for vacated pixels, kills trailing/ghosting)

All branchless: `step()` selects the rate, `mix()` applies it. No GPU warp divergence.

#### Stage 1.5: Morphological Closing

**Problem:** The mask has small holes (e.g., patterned clothing classified as background) and jagged edges from the low-resolution model.

**Solution:** Dilate (max of 3x3 neighborhood) then erode (min of 3x3). This "closing" operation fills holes smaller than 1 pixel and smooths jagged boundaries. Operates at mask resolution (256x256 = 65K pixels), not full frame (1280x720 = 921K pixels), so it's cheap.

**Math:** `result = mix(max(result, sample), min(result, sample), operation)` — branchless dilate/erode via the `operation` uniform (0.0=dilate, 1.0=erode).

#### Stage 2: Joint Bilateral Upsample

**Problem:** The mask is 256x256 but the output is 1280x720. Naive bilinear upsampling creates blocky stair-step edges that don't align with the actual person boundary in the camera frame.

**Solution:** Joint bilateral filter — a 5x5 kernel where each sample's weight depends on both spatial distance and color similarity to the center pixel in the full-resolution camera frame. Mask edges "snap" to actual RGB boundaries.

**Math:** For each of 25 kernel samples:
```
spatialWeight = exp(-spatialDist² / (2σ_spatial²))
rangeWeight   = exp(-colorDist² / (2σ_range²))
weight        = spatialWeight × rangeWeight
```

**Chroma-weighted distance:** Standard RGB distance treats `(1.0, 0.95, 0.9)` (warm skin) and `(1.0, 1.0, 1.0)` (white wall) as similar. We separate luminance and chroma:
```
lumDiff   = dot(colorDiff, [0.299, 0.587, 0.114])
chromaDiff = colorDiff - lumDiff
dist²     = lumDiff² + dot(chromaDiff, chromaDiff) × 3.0
```
The 3x chroma weight amplifies the skin-vs-white distinction, preventing the mask from bleeding into white backgrounds.

**Optimization:** Spatial distances are precomputed as `const float[25]` arrays. Reciprocals (`1/(2σ²)`) are hoisted outside the loop. No division in the inner loop.

#### Stage 3: Edge Feathering

**Problem:** Even after bilateral upsampling, mask edges can appear hard/aliased. But we don't want to blur the entire mask — that would soften interior foreground.

**Solution:** Detect edge pixels by checking the mask gradient (max absolute difference across 8 neighbors). If `maxDiff > 0.02`, apply a 5x5 Gaussian blur. If not, early-exit — the pixel passes through untouched.

**Performance:** ~90% of pixels are pure foreground or pure background with zero gradient. The early-exit skips the 25-tap blur for these pixels. Since they're spatially coherent (large regions of 0.0 or 1.0), this is a well-predicted branch with massive savings.

#### Stage 3.5: Mask Erosion

**Problem:** The outermost ring of "foreground" pixels often contains blended color from the real background (the model classified them as foreground, but they're actually mixed pixels at the boundary).

**Solution:** A 0.5px erode pass at full resolution shrinks the mask inward by half a pixel. This cuts off the contaminated edge pixels. Combined with foreground recovery in Stage 4, this eliminates halos without losing significant edge detail.

#### Stage 4: Compositing (with Foreground Recovery)

The most sophisticated stage, combining three techniques in a single shader pass:

**1. Edge-Adaptive Sharpening**

The camera frame's RGB gradient tells us about edge clarity:
```
edgeStrength = dot(dx, dx) + dot(dy, dy)
sharpness    = smoothstep(0.001, 0.02, edgeStrength)
lo           = mix(0.15, 0.35, sharpness)   // mask threshold low
hi           = mix(0.85, 0.65, sharpness)   // mask threshold high
mask         = smoothstep(lo, hi, rawMask)
```

Where the gradient is strong (sharp edge like shoulder against background), the smoothstep range narrows (0.35–0.65), creating a crisp transition. Where the gradient is weak (ambiguous hair wisps), the range widens (0.15–0.85), keeping a soft transition that preserves detail.

**2. Closed-Form Alpha Matting**

In the mask transition zone (0.02–0.98), a 13-sample cross-shaped kernel estimates nearby foreground color `F` and background color `B`. The matting equation from Levin et al.:

```
alpha = dot(I - B, F - B) / dot(F - B, F - B)
```

This projects the observed pixel color `I` onto the line between `F` and `B`, giving mathematically correct alpha. It handles semi-transparent edges (hair, motion blur) where the model's binary mask is insufficient.

**Color separation gate:** When foreground and background colors are too similar (e.g., skin near a beige wall), the matting equation becomes numerically unstable. A perceptual distance gate disables it:
```
perceptualDenom = lumDiff² + dot(chromaDiff, chromaDiff) × 3.0
colorSeparation = smoothstep(0.02, 0.08, perceptualDenom)
```
When `colorSeparation ≈ 0`, matting is bypassed and the raw mask is used instead.

**3. Foreground Recovery (VFX-standard)**

The camera pixel at a boundary is contaminated by the *original* background:
```
I_observed = F_true × alpha + B_old × (1 - alpha)
```

We want the output with the *new* background:
```
output = F_true × alpha + B_new × (1 - alpha)
```

Subtracting: `output = I_observed + (B_new - B_old) × (1 - alpha)`

This single equation mathematically removes the old background's color contribution and replaces it with the new one. No heuristic color sampling, no inward-gradient tricks — pure math. Especially effective against white/light backgrounds where color bleed is most visible.

**Blend factor:** Active across a wide mask range `smoothstep(0.02, 0.15, rawMask) × (1.0 - smoothstep(0.9, 1.0, rawMask))` to catch misclassified edge pixels that the model calls "foreground" but still contain background color.

#### Stage 5: Light Wrap

**Problem:** Even with perfect segmentation, the person looks "cut out and pasted on" because there's no light interaction between foreground and background.

**Solution:** Subtle background color spill onto the narrow edge band of the person's silhouette:
```
edgeMask = smoothstep(0.25, 0.45, mask) × (1.0 - smoothstep(0.55, 0.75, mask))
output   = mix(composite, background, edgeMask × 0.06)
```

Only active on the [0.25–0.75] mask range at 6% strength. This simulates real-world light wrapping where the background illumination slightly affects the subject's edges.

#### Stage 6: Auto-Frame (optional)

**Problem:** In video calls, the person is often off-center or too small in frame.

**Solution:** GPU crop shader samples a sub-region of the composited output:
```
uv = cropOffset + texCoord × cropSize
```

The crop rectangle is computed from the segmentation mask's bounding box with:
- **Distance-adaptive zoom:** `zoom = targetFill / actualFill` (target 90% fill, min 1.2x, max 1.5x)
- **Vertical centering:** `vertOffset = 0.55 + (1 - fillRatio) × 0.03` (head higher in frame when further away)
- **Exponential smoothing:** 0.75 factor (~100ms tracking) prevents camera swim

Zero CPU overhead — no canvas copy, just a single extra GPU draw call.

### Why it's fast

| Technique | Savings |
|-----------|---------|
| 30fps render loop matched to model — zero interpolation overhead | No wasted frames or stale masks |
| ROI cropping — person bbox from previous mask crops model input | ~2x more resolution on edges without larger model |
| CPU inference (WASM+SIMD+XNNPACK), GPU free for rendering | No GPU contention |
| 256x256 model input | 14x fewer pixels than 1280x720 |
| Background blur at half resolution with multi-pass Gaussian | 4x fewer blur pixels |
| All post-processing in WebGL2 fragment shaders | Zero CPU pixel readback |
| Auto-frame via GPU crop shader (no CPU canvas copy) | Zero-cost centering/zoom |
| Branchless shaders (step/mix instead of if/else) | No warp divergence |
| Precomputed distance tables and reciprocals in hot loops | No division in inner loops |
| Constant vec2/float arrays for kernel offsets | No int-to-float conversion |
| Early-exit in feather shader (90%+ non-edge pixels skip 25-tap blur) | Coherent branch, massive savings |
| `inversesqrt()` instead of `length()`+division | Hardware-accelerated normalization |
| `dot()` instead of `length()` for comparisons | Avoids sqrt where magnitude isn't needed |

### Why it looks good

| Technique | Effect |
|-----------|--------|
| ROI cropping (person bounding box) | ~2x model resolution on person edges — sharper boundaries |
| Temporal smoothing with hysteresis | No flickering — fast appear, slow disappear |
| Motion-aware blend rates | Fast movement tracks instantly, still areas maximally smooth |
| Morphological closing (dilate→erode) | Fills mask holes, smooths jagged edges |
| Chroma-weighted bilateral upsample | Low-res mask edges snap to RGB boundaries; 3x chroma weight distinguishes skin from white backgrounds |
| Selective edge feathering | Soft edges without blurring the interior |
| Mask erosion (anti-halo) | Cuts contaminated edge pixels to eliminate halo |
| Foreground recovery (`I + (newBg-B)*(1-α)`) | Mathematically removes original background — no color bleed on white/light backgrounds |
| Alpha matting (closed-form) | Per-pixel hair strand isolation using `alpha = (I-B)/(F-B)` |
| Color separation gating | Disables matting where fg/bg colors are similar (prevents artifacts) |
| Edge-adaptive sharpening | Sharp edges at clear boundaries, soft at ambiguous hair |
| Light wrapping | Background light spills onto edges — no "cutout" look |

## Adaptive Quality

On by default. Monitors frame times and auto-adjusts:

| Tier | Model | FPS | Morphology | Light Wrap |
|------|-------|-----|------------|------------|
| ultra | 256x256 | 30 | yes | yes |
| high | 256x256 | 24 | yes | yes |
| medium | 256x256 | 12 | yes | yes |
| low | 160x160 | 10 | no | no |
| minimal | 160x160 | 8 | no | no |

Downgrades fast (2 bad windows). Upgrades slow (5 good windows). 3 critical frames (>40ms) triggers immediate downgrade.

```ts
// Disable
new SegmentationProcessor({ adaptive: false, quality: 'high' });

// Monitor
new SegmentationProcessor({
  adaptiveConfig: {
    onQualityChange: (level, reason) => console.log(level.label, reason),
  },
});

// Lock after user selection
processor.getAdaptiveController()?.lock();
```

## Auto-Framing

Google Meet-style centering that keeps the subject in frame:

```ts
const processor = new SegmentationProcessor({
  autoFrame: { enabled: true, continuous: true },
});

// Or toggle at runtime
processor.setAutoFrame(true);

// Read crop rect for your renderer
const crop = processor.getCropRect();
// { x: 0.1, y: 0.05, width: 0.8, height: 0.9, zoom: 1.2 }
```

Uses segmentation mask to derive person bounding box, applies exponential smoothing for stable tracking. Distance-adaptive zoom (further = more zoom, targetFill 90%) keeps the subject well-framed. Rendered entirely on GPU via a crop shader — zero CPU overhead.

## Performance

| Device | Tier | Model | Pipeline | Total |
|--------|------|-------|----------|-------|
| MacBook Pro M-series | ultra | ~13ms | ~0.2ms | ~13ms |
| MacBook Air M2 | high | ~8ms | ~0.3ms | ~8ms |
| Mid-range Windows | medium | ~12ms | ~0.5ms | ~12ms |
| Chromebook | low | ~18ms | ~1ms | ~19ms |

Note: Pipeline time is GPU command issue time (CPU-side). Actual GPU execution overlaps with CPU work, so total frame time is dominated by model inference.

```ts
const m = processor.getMetrics();
// { fps, modelFps, modelInferenceMs, pipelineMs, totalFrameMs, skippedFrames }
```

## API

### `SegmentationProcessor`

```ts
new SegmentationProcessor({
  backgroundMode: 'blur',           // 'blur' | 'image' | 'color' | 'none'
  blurRadius: 12,                   // 4-24
  backgroundColor: '#1a1a2e',       // hex
  backgroundImage: null,            // HTMLImageElement
  quality: 'medium',                // 'low' | 'medium' | 'high'
  adaptive: true,                   // auto quality scaling
  useWorker: true,                  // off-main-thread inference (0ms main thread)
  modelFps: 30,                     // inference rate (matched to render)
  debug: false,                     // log metrics
  autoFrame: { enabled: false },    // auto-centering
  modelConfig: { delegate: 'CPU' }, // 'CPU' | 'GPU'
})
```

| Method | Description |
|--------|-------------|
| `toLiveKitProcessor()` | Official LiveKit `TrackProcessor` (uses `processedTrack`) |
| `createProcessedTrack(track)` | Standalone `MediaStreamTrack` (non-LiveKit) |
| `setBackgroundMode(mode)` | Switch mode |
| `setBackgroundColor(hex)` | Set color |
| `setBackgroundImage(img)` | Set image |
| `setBlurRadius(n)` | Adjust blur |
| `setQuality(preset)` | Manual quality |
| `setAutoFrame(on, continuous?)` | Toggle auto-centering |
| `getMetrics()` | Performance data |
| `getCropRect()` | Auto-frame crop |
| `getAdaptiveController()` | Quality controller |
| `getAutoFramer()` | Auto-frame controller |
| `destroy()` | Release resources |

### `PostProcessingPipeline`

Low-level WebGL2 pipeline for custom integrations:

```ts
import { PostProcessingPipeline } from 'segmo';

const pipeline = new PostProcessingPipeline({
  width: 1280, height: 720,
  maskWidth: 256, maskHeight: 256,
  backgroundMode: 'blur',
  blurRadius: 12,
  appearRate: 0.75,
  disappearRate: 0.35,
  featherRadius: 3.0,
  rangeSigma: 0.1,
  morphology: true,
  lightWrap: true,
});

const output = pipeline.process(cameraFrame, maskData, motionMap);
const interpolated = pipeline.processInterpolated(cameraFrame);
```

## WebGL2 Shader Pipeline

All shaders are optimized for real-time GPU execution:

- **Branchless operations**: `step()`, `mix()`, `smoothstep()` instead of `if/else` to avoid warp divergence
- **Precomputed tables**: Constant `float[25]` and `vec2[25]` arrays for spatial distances and kernel offsets
- **Hoisted invariants**: `texelSize * radius`, reciprocals, and other loop-invariant values computed outside loops
- **Squared comparisons**: `dot(v,v)` instead of `length(v)` to avoid unnecessary `sqrt()`
- **`inversesqrt()`**: Hardware-optimized normalization without separate `length()` + division
- **Cross-shaped sampling**: Alpha matting uses a 13-sample cross pattern (wider directional reach, fewer samples than a full grid)
- **Color separation gating**: Matting auto-disables via `smoothstep()` when fg/bg colors are similar

## Demos

```bash
npx serve .
# Open http://localhost:3000/demo/standalone
```

The standalone demo includes:
- Camera selection
- All background modes (blur, color, image, none)
- Quality controls (Auto, Low, Med, High)
- Background image presets (procedurally generated)
- Custom image upload
- Auto-frame toggle
- Real-time performance metrics (FPS, model FPS, pipeline time, quality tier)

## Browser Support

Chrome 97+, Edge 97+, Firefox 105+, Safari 16.4+

Requires: WebGL2, `EXT_color_buffer_float`, `OES_texture_float_linear`, MediaPipe WASM

## License

MIT
