/**
 * Example: segmo standalone (no LiveKit)
 *
 * Works with any MediaStreamTrack — useful for custom WebRTC apps,
 * recording, or non-LiveKit video conferencing.
 *
 * Install:
 *   npm install segmo
 */

import { SegmentationProcessor, DiagnosticEvent } from 'segmo';

async function main() {
  // 0. Check browser capabilities
  const caps = SegmentationProcessor.checkCapabilities();
  if (!caps.supported) {
    document.body.textContent = 'Browser not supported: ' + caps.unsupportedReasons.join('; ');
    return;
  }

  // 1. Get camera
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, frameRate: 30 },
  });
  const inputTrack = stream.getVideoTracks()[0];

  // 2. Create processor with diagnostics
  const processor = new SegmentationProcessor({
    backgroundMode: 'blur',
    blurRadius: 14,
    adaptive: true,
    quality: 'medium',
    useWorker: true, // off-main-thread inference (0ms main thread)

    // Diagnostics — collects periodic summaries, your app decides where to send them
    diagnosticsLevel: 'summary',
    diagnosticsIntervalMs: 10000,
    diagnosticsIncludeImage: false,
    clientId: 'user-123', // unique identifier for this client/session
    onDiagnostic: (event: DiagnosticEvent) => {
      console.log(`[diag] ${event.type}`, event.data);

      // Send to your telemetry endpoint
      fetch('/api/telemetry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(event),
      });
    },
  });

  // 3. Create processed track (Insertable Streams API)
  const outputTrack = await processor.createProcessedTrack(
    inputTrack as MediaStreamVideoTrack,
  );

  // 4. Display
  const video = document.querySelector('video')!;
  video.srcObject = new MediaStream([outputTrack]);
  await video.play();

  // 5. Switch backgrounds dynamically
  document.getElementById('btn-blur')?.addEventListener('click', () => {
    processor.setBackgroundMode('blur');
    processor.setBlurRadius(14);
  });

  document.getElementById('btn-color')?.addEventListener('click', () => {
    processor.setBackgroundMode('color');
    processor.setBackgroundColor('#1a1a2e');
  });

  document.getElementById('btn-image')?.addEventListener('click', () => {
    const img = new Image();
    img.src = '/backgrounds/mountains.jpg';
    img.onload = () => processor.setBackgroundImage(img);
  });

  document.getElementById('btn-off')?.addEventListener('click', () => {
    processor.setBackgroundMode('none');
  });

  // 5b. Fixed background with auto-frame
  // When backgroundFixed is true, the virtual background stays stationary
  // while auto-frame pans/zooms only the subject — no parallax effect.
  document.getElementById('btn-fixed-bg')?.addEventListener('click', () => {
    const fixedProcessor = new SegmentationProcessor({
      backgroundMode: 'image',
      useWorker: true,
      backgroundFixed: true,
      autoFrame: { enabled: true, continuous: true },
    });
    const img = new Image();
    img.src = '/backgrounds/mountains.jpg';
    img.onload = () => fixedProcessor.setBackgroundImage(img);
  });

  // 6. On-demand snapshot (e.g. "Report Issue" button)
  document.getElementById('btn-report')?.addEventListener('click', () => {
    const snapshot = processor.exportDiagnosticSnapshot();
    console.log('[diag] snapshot', snapshot);

    // Send to your telemetry endpoint
    // fetch('/api/telemetry/snapshot', { method: 'POST', body: JSON.stringify(snapshot) });
  });

  // 7. Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    processor.destroy();
    inputTrack.stop();
  });
}

main().catch(console.error);
