/**
 * Type declarations for WebCodecs Insertable Streams
 * (MediaStreamTrackProcessor / MediaStreamTrackGenerator)
 *
 * These APIs are available in Chrome 94+ but not in TypeScript's default DOM lib.
 */

declare class MediaStreamTrackProcessor {
  constructor(init: { track: MediaStreamTrack });
  readonly readable: ReadableStream<VideoFrame>;
}

declare class MediaStreamTrackGenerator extends MediaStreamTrack {
  constructor(init: { kind: string });
  readonly writable: WritableStream<VideoFrame>;
}

interface MediaStreamVideoTrack extends MediaStreamTrack {
  kind: 'video';
}
