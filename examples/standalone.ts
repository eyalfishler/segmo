/**
 * Example: segmo standalone (no LiveKit)
 *
 * Works with any MediaStreamTrack — useful for custom WebRTC apps,
 * recording, or non-LiveKit video conferencing.
 *
 * Install:
 *   npm install segmo
 */

import { SegmentationProcessor } from 'segmo';

async function main() {
  // 1. Get camera
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, frameRate: 30 },
  });
  const inputTrack = stream.getVideoTracks()[0];

  // 2. Create processor
  const processor = new SegmentationProcessor({
    backgroundMode: 'blur',
    blurRadius: 14,
    adaptive: true,
    quality: 'medium',
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

  // 6. Monitor performance
  setInterval(() => {
    const m = processor.getMetrics();
    console.log(
      `FPS: ${m.fps} | Model: ${m.modelFps}fps @ ${m.modelInferenceMs.toFixed(1)}ms | ` +
      `Total: ${m.totalFrameMs.toFixed(1)}ms`,
    );
  }, 3000);

  // 7. Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    processor.destroy();
    inputTrack.stop();
  });
}

main().catch(console.error);
