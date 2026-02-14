# Changelog

## 0.1.3

### Features
- Add `backgroundFixed` option — keeps virtual background stationary during auto-frame crop instead of panning/zooming with the subject (parallax). Applies to image and color background modes.

## 0.1.2

### Bug Fixes
- Fix mask boundary artifacts caused by motion compensation shift shader zeroing out edge pixels on interpolated frames — now clamps instead of discarding OOB samples
- Pad mask edges (4 rows/cols) before GPU upload to prevent model low-confidence boundary values from being amplified by bilateral/feathering/erosion kernels
- Start adaptive quality at ultra (tier 0) on init to eliminate ~5 second startup period where movement caused visible blur artifacts — downgrades fast (2 bad windows) if device can't sustain it
- Coerce `init()` width/height to positive integers for OffscreenCanvas safety

### Docs
- Clarify LiveKit processor usage in README (`toLiveKitProcessor()` guidance)

## 0.1.0

Initial release.

### GPU Pipeline (7 stages)
- Temporal smoothing with motion-aware hysteresis (asymmetric appear/disappear rates)
- Morphological closing (dilate→erode) at mask resolution for hole-filling
- Chroma-weighted joint bilateral upsample (5x5 kernel, 3x chroma weight for skin-vs-white)
- Selective edge feathering (5x5 Gaussian, early-exit for 90%+ non-edge pixels)
- 0.5px mask erosion at full resolution (anti-halo)
- Compositing with closed-form alpha matting, edge-adaptive sharpening, and foreground recovery
- Light wrap (6% background spill on edge band)

### Segmentation Model
- MediaPipe selfie segmenter (256x256 square, CPU WASM+SIMD, float16)
- ROI cropping from previous mask bbox (~2x edge resolution gain)
- Dead zone smoothing (3% position, 1.5% size, 50% EMA) prevents crop jitter
- Cached bbox computation during ROI mapping (eliminates separate 65K pixel scan)

### Motion Compensation
- 3-zone centroid velocity tracking (top/mid/bottom Y bands, EMA α=0.8)
- GPU mask shift shader translates stale mask on interpolated frames
- Adaptive model rate: up to 4x faster during movement (capped at 16ms)
- Dead zone (0.3% of frame width) filters noise-level velocity

### Background Modes
- Blur: 3-pass separable Gaussian at half resolution (13-tap kernel)
- Image replacement
- Solid color
- None (passthrough)
- All switchable at runtime without re-initialization

### Adaptive Quality
- 5 tiers: ultra (30fps), high (24fps), medium (12fps), low (10fps), minimal (8fps)
- Low/minimal drop to 160x160 model and disable morphology + light wrap
- Asymmetric response: downgrades after 2 bad windows, upgrades after 5 good windows
- 3 critical frames (>40ms) triggers immediate downgrade
- 1-second cooldown between adjustments

### Auto-Framing
- Mask-based person bounding box with weighted centroid
- Distance-adaptive zoom (target 80% fill, 1.1x–4.4x range)
- Head-aware vertical framing (15% headroom)
- Exponential smoothing (0.75 factor) for stable tracking
- GPU crop shader (zero CPU overhead)

### Web Worker Support
- Off-main-thread inference via inline Blob URL worker (no separate file)
- Zero-copy transfer of ImageBitmap (in) and Float32Array buffers (out)
- Automatic fallback to main thread on worker init failure
- ROI cropping and bbox computation replicated in worker

### Integration
- LiveKit `TrackProcessor` interface via `toLiveKitProcessor()`
- Standalone `MediaStreamTrack` via `createProcessedTrack()`
- Low-level `PostProcessingPipeline` for custom integrations
- All types and interfaces exported
