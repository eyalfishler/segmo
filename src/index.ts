/**
 * segmo
 *
 * Google Meet-quality background segmentation for video conferencing.
 *
 * Quick start:
 *
 * ```ts
 * import { SegmentationProcessor } from 'segmo';
 *
 * const processor = new SegmentationProcessor({
 *   backgroundMode: 'blur',
 * });
 *
 * // With LiveKit
 * const track = await createLocalVideoTrack();
 * await track.setProcessor(processor.toLiveKitProcessor());
 *
 * // Or standalone
 * const outputTrack = await processor.createProcessedTrack(inputTrack);
 * ```
 */

export { SegmentationProcessor } from './processor';
export type { SegmentationProcessorOptions, BackgroundMode } from './processor';

export { PostProcessingPipeline } from './pipeline';
export type { PipelineOptions } from './pipeline';

export { SegmentationModel } from './model';
export type { ModelConfig, CropRegion } from './model';

export { AdaptiveQualityController } from './adaptive';
export type { AdaptiveConfig, QualityLevel } from './adaptive';

export { AutoFramer } from './autoframe';
export type { AutoFrameConfig, CropRect } from './autoframe';

export { ModelWorkerClient } from './model-worker';
export type { WorkerMaskResult } from './model-worker';
