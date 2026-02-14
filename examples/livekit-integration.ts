/**
 * Example: segmo + LiveKit
 *
 * Production-ready integration showing background mode switching,
 * adaptive quality, auto-framing, and performance monitoring.
 *
 * Install:
 *   npm install segmo livekit-client
 */

import { SegmentationProcessor, type BackgroundMode } from 'segmo';
import {
  Room,
  RoomEvent,
  Track,
  createLocalVideoTrack,
  type LocalVideoTrack,
  type RemoteTrackPublication,
  type RemoteParticipant,
} from 'livekit-client';

// ============================================================
// 1. Setup
// ============================================================

const processor = new SegmentationProcessor({
  backgroundMode: 'blur',
  blurRadius: 12,
  adaptive: true,                // auto-adjusts quality to device
  useWorker: true,               // off-main-thread inference (0ms main thread)
  debug: false,                  // set true to log metrics
  backgroundFixed: false,        // true = bg stays fixed during auto-frame (no parallax)
  autoFrame: { enabled: false }, // enable later if wanted
  adaptiveConfig: {
    onQualityChange: (level, reason) => {
      console.log(`[segmo] Quality → ${level.label}: ${reason}`);
      updateUI({ quality: level.label });
    },
  },
});

const room = new Room();

// ============================================================
// 2. Connect & Publish
// ============================================================

async function joinRoom(serverUrl: string, token: string) {
  // Create video track
  const videoTrack = await createLocalVideoTrack({
    resolution: { width: 1280, height: 720, frameRate: 30 },
  });

  // Attach segmo processor (implements official TrackProcessor interface via processedTrack)
  await videoTrack.setProcessor(processor.toLiveKitProcessor());

  // Connect and publish
  await room.connect(serverUrl, token);
  await room.localParticipant.publishTrack(videoTrack);

  // Attach local preview
  const localVideo = document.getElementById('local-video') as HTMLVideoElement;
  videoTrack.attach(localVideo);

  // Handle remote participants
  room.on(RoomEvent.TrackSubscribed, handleRemoteTrack);
  room.on(RoomEvent.Disconnected, handleDisconnect);
}

// ============================================================
// 3. Background Controls
// ============================================================

function setBlurBackground(radius = 12) {
  processor.setBackgroundMode('blur');
  processor.setBlurRadius(radius);
}

function setImageBackground(url: string) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = url;
  img.onload = () => processor.setBackgroundImage(img);
  img.onerror = () => console.error('Failed to load background image:', url);
}

function setColorBackground(hex: string) {
  processor.setBackgroundMode('color');
  processor.setBackgroundColor(hex);
}

function disableBackground() {
  processor.setBackgroundMode('none');
}

// ============================================================
// 4. Quality Controls
// ============================================================

/** Let adaptive controller manage quality automatically */
function enableAdaptiveQuality() {
  processor.getAdaptiveController()?.unlock();
}

/** User manually picks quality (locks adaptive) */
function setManualQuality(level: 'low' | 'medium' | 'high') {
  processor.setQuality(level);
  processor.getAdaptiveController()?.lock();
}

/** Log performance metrics */
function logPerformance() {
  const m = processor.getMetrics();
  console.table({
    fps: m.fps,
    modelFps: m.modelFps,
    modelMs: m.modelInferenceMs.toFixed(1),
    pipelineMs: m.pipelineMs.toFixed(1),
    totalMs: m.totalFrameMs.toFixed(1),
    skipped: m.skippedFrames,
  });
}

// ============================================================
// 5. Auto-Framing
// ============================================================

function enableAutoFrame() {
  processor.setAutoFrame(true, true);
}

function disableAutoFrame() {
  processor.setAutoFrame(false);
}

// ============================================================
// 6. Cleanup
// ============================================================

async function leaveRoom() {
  processor.destroy();
  await room.disconnect();
}

// ============================================================
// Helpers
// ============================================================

function handleRemoteTrack(
  track: any,
  publication: RemoteTrackPublication,
  participant: RemoteParticipant,
) {
  if (track.kind === Track.Kind.Video) {
    const el = document.createElement('video');
    el.id = `remote-${participant.identity}`;
    el.autoplay = true;
    track.attach(el);
    document.getElementById('remote-videos')?.appendChild(el);
  }
}

function handleDisconnect() {
  processor.destroy();
  document.querySelectorAll('[id^="remote-"]').forEach(el => el.remove());
}

function updateUI(state: { quality?: string }) {
  const badge = document.getElementById('quality-badge');
  if (badge && state.quality) badge.textContent = state.quality;
}

// ============================================================
// Wire to your UI
// ============================================================

// Join
document.getElementById('btn-join')?.addEventListener('click', () => {
  const url = (document.getElementById('server-url') as HTMLInputElement).value;
  const token = (document.getElementById('token') as HTMLInputElement).value;
  joinRoom(url, token);
});

// Background mode buttons
document.getElementById('btn-blur')?.addEventListener('click', () => setBlurBackground());
document.getElementById('btn-color')?.addEventListener('click', () => setColorBackground('#1a1a2e'));
document.getElementById('btn-image')?.addEventListener('click', () => setImageBackground('/bg/office.jpg'));
document.getElementById('btn-none')?.addEventListener('click', () => disableBackground());

// Quality
document.getElementById('btn-auto')?.addEventListener('click', () => enableAdaptiveQuality());
document.getElementById('btn-low')?.addEventListener('click', () => setManualQuality('low'));
document.getElementById('btn-med')?.addEventListener('click', () => setManualQuality('medium'));
document.getElementById('btn-high')?.addEventListener('click', () => setManualQuality('high'));

// Blur slider
document.getElementById('blur-slider')?.addEventListener('input', (e) => {
  processor.setBlurRadius(parseInt((e.target as HTMLInputElement).value));
});

// Auto-frame toggle
document.getElementById('btn-autoframe')?.addEventListener('click', () => {
  const btn = document.getElementById('btn-autoframe') as HTMLButtonElement;
  const isOn = btn.dataset.state === 'on';
  if (isOn) { disableAutoFrame(); btn.dataset.state = 'off'; btn.textContent = 'Auto-Frame: Off'; }
  else { enableAutoFrame(); btn.dataset.state = 'on'; btn.textContent = 'Auto-Frame: On'; }
});

// Leave
document.getElementById('btn-leave')?.addEventListener('click', () => leaveRoom());

// Performance logging
setInterval(logPerformance, 5000);
