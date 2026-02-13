/**
 * Adaptive Quality Controller
 *
 * Monitors frame processing times and automatically adjusts quality settings
 * to maintain smooth output. This is what Google Meet does — they call it
 * "adaptive processing" and it's why Meet works on both a Chromebook and
 * a gaming PC.
 *
 * Strategy:
 * - Track rolling average of frame times
 * - If consistently over budget → downgrade (model fps, resolution, feather quality)
 * - If consistently under budget → upgrade (but with hysteresis to avoid oscillation)
 * - React quickly to drops, slowly to improvements (asymmetric response)
 *
 * Degradation order (least noticeable first):
 * 1. Reduce model FPS (30→15→10)
 * 2. Reduce model resolution (256×144 → 160×96)
 * 3. Disable light wrap
 * 4. Reduce feather quality
 * 5. Reduce blur passes
 * 6. Last resort: disable entirely
 */

export interface AdaptiveConfig {
  /** Target frame time in ms (default: 28ms for 30fps with 2ms headroom) */
  targetFrameTimeMs?: number;
  /** Frame time above this triggers immediate downgrade (default: 40ms) */
  criticalFrameTimeMs?: number;
  /** Number of frames to average over (default: 30) */
  windowSize?: number;
  /** Consecutive good windows before upgrading (default: 5) — prevents oscillation */
  upgradeThreshold?: number;
  /** Consecutive bad windows before downgrading (default: 2) — react fast to drops */
  downgradeThreshold?: number;
  /** Enable logging of quality changes (default: false) */
  debug?: boolean;
  /** Callback when quality changes */
  onQualityChange?: (level: QualityLevel, reason: string) => void;
}

export interface QualityLevel {
  /** Current quality tier (0 = highest, 5 = lowest/disabled) */
  tier: number;
  /** Human-readable label */
  label: string;
  /** Model FPS */
  modelFps: number;
  /** Model input width */
  modelWidth: number;
  /** Model input height */
  modelHeight: number;
  /** Edge feather radius */
  featherRadius: number;
  /** Bilateral range sigma */
  rangeSigma: number;
  /** Light wrap enabled */
  lightWrap: boolean;
  /** Background blur radius */
  blurRadius: number;
  /** Temporal appear rate */
  appearRate: number;
  /** Temporal disappear rate */
  disappearRate: number;
}

// Quality tiers ordered from highest to lowest
// Degradation order chosen by what's least noticeable to users
const QUALITY_TIERS: QualityLevel[] = [
  {
    tier: 0,
    label: 'ultra',
    modelFps: 30,
    modelWidth: 256,
    modelHeight: 256,
    featherRadius: 4.0,
    rangeSigma: 0.08,
    lightWrap: true,
    blurRadius: 12,
    appearRate: 0.7,
    disappearRate: 0.3,
  },
  {
    tier: 1,
    label: 'high',
    modelFps: 24,
    modelWidth: 256,
    modelHeight: 256,
    featherRadius: 3.0,
    rangeSigma: 0.1,
    lightWrap: true,
    blurRadius: 12,
    appearRate: 0.75,
    disappearRate: 0.35,
  },
  {
    tier: 2,
    label: 'medium',
    modelFps: 12,
    modelWidth: 256,
    modelHeight: 256,
    featherRadius: 2.5,
    rangeSigma: 0.12,
    lightWrap: true,
    blurRadius: 10,
    appearRate: 0.8,
    disappearRate: 0.4,
  },
  {
    tier: 3,
    label: 'low',
    modelFps: 10,
    modelWidth: 160,
    modelHeight: 160,
    featherRadius: 2.0,
    rangeSigma: 0.15,
    lightWrap: false,
    blurRadius: 8,
    appearRate: 0.85,
    disappearRate: 0.45,
  },
  {
    tier: 4,
    label: 'minimal',
    modelFps: 8,
    modelWidth: 160,
    modelHeight: 160,
    featherRadius: 1.5,
    rangeSigma: 0.2,
    lightWrap: false,
    blurRadius: 6,
    appearRate: 0.9,
    disappearRate: 0.5,
  },
];

export class AdaptiveQualityController {
  private config: Required<AdaptiveConfig>;
  private currentTier = 1; // Start at 'high', not 'ultra'
  private frameTimes: number[] = [];
  private windowIndex = 0;
  private consecutiveGoodWindows = 0;
  private consecutiveBadWindows = 0;
  private criticalFrameCount = 0;
  private totalFrames = 0;
  private lastAdjustmentTime = 0;
  private locked = false;

  // Callbacks to apply quality changes
  private applyCallbacks: Array<(level: QualityLevel) => void> = [];

  constructor(config: AdaptiveConfig = {}) {
    this.config = {
      targetFrameTimeMs: config.targetFrameTimeMs ?? 28,
      criticalFrameTimeMs: config.criticalFrameTimeMs ?? 40,
      windowSize: config.windowSize ?? 30,
      upgradeThreshold: config.upgradeThreshold ?? 5,
      downgradeThreshold: config.downgradeThreshold ?? 2,
      debug: config.debug ?? false,
      onQualityChange: config.onQualityChange ?? (() => { }),
    };

    this.frameTimes = new Array(this.config.windowSize).fill(0);
  }

  /**
   * Register a callback that gets called when quality changes.
   * Used by the processor to apply new settings.
   */
  onApply(callback: (level: QualityLevel) => void): void {
    this.applyCallbacks.push(callback);
  }

  /**
   * Report a frame's processing time. Call this every frame.
   *
   * @param frameTimeMs - Total time to process this frame (model + pipeline)
   */
  reportFrame(frameTimeMs: number): void {
    this.frameTimes[this.windowIndex] = frameTimeMs;
    this.windowIndex = (this.windowIndex + 1) % this.config.windowSize;
    this.totalFrames++;

    // Track critical frames (immediate response)
    if (frameTimeMs > this.config.criticalFrameTimeMs) {
      this.criticalFrameCount++;

      // 3 critical frames in a row = immediate downgrade
      if (this.criticalFrameCount >= 3) {
        this.downgrade('critical frame times');
        this.criticalFrameCount = 0;
        return;
      }
    } else {
      this.criticalFrameCount = 0;
    }

    // Evaluate at window boundaries
    if (this.totalFrames % this.config.windowSize === 0) {
      this.evaluateWindow();
    }
  }

  /** Get the current quality level */
  getCurrentLevel(): QualityLevel {
    return QUALITY_TIERS[this.currentTier];
  }

  /** Get current tier index */
  getCurrentTier(): number {
    return this.currentTier;
  }

  /**
   * Lock quality at current level (prevents auto-adjustment).
   * Useful when user explicitly selects a quality.
   */
  lock(): void {
    this.locked = true;
  }

  /** Unlock quality for auto-adjustment */
  unlock(): void {
    this.locked = false;
  }

  /** Force a specific quality tier */
  setTier(tier: number): void {
    const clamped = Math.max(0, Math.min(tier, QUALITY_TIERS.length - 1));
    if (clamped !== this.currentTier) {
      this.currentTier = clamped;
      this.applyCurrentLevel('manual override');
    }
  }

  /** Reset controller state (e.g., after tab becomes visible again) */
  reset(): void {
    this.frameTimes.fill(0);
    this.windowIndex = 0;
    this.consecutiveGoodWindows = 0;
    this.consecutiveBadWindows = 0;
    this.criticalFrameCount = 0;
    this.totalFrames = 0;
  }

  /**
   * Run initial benchmark to determine starting quality.
   * Call this during init with a few test frames.
   *
   * @param sampleFrameTimeMs - Average frame time from benchmark
   */
  calibrateFromBenchmark(sampleFrameTimeMs: number): void {
    const target = this.config.targetFrameTimeMs;

    if (sampleFrameTimeMs < target * 0.5) {
      // Very fast device — start at ultra
      this.currentTier = 0;
    } else if (sampleFrameTimeMs < target * 0.8) {
      // Fast device — start at high
      this.currentTier = 1;
    } else if (sampleFrameTimeMs < target) {
      // Normal device — start at medium
      this.currentTier = 2;
    } else if (sampleFrameTimeMs < target * 1.5) {
      // Slow device — start at low
      this.currentTier = 3;
    } else {
      // Very slow — start at minimal
      this.currentTier = 4;
    }

    this.log(`Calibrated to tier ${this.currentTier} (${this.getCurrentLevel().label}) ` +
      `from benchmark ${sampleFrameTimeMs.toFixed(1)}ms`);
    this.applyCurrentLevel('initial calibration');
  }

  // === Private ===

  private evaluateWindow(): void {
    if (this.locked) return;

    // Cooldown: don't adjust more than once per second
    const now = performance.now();
    if (now - this.lastAdjustmentTime < 1000) return;

    const avg = this.getAverageFrameTime();
    const target = this.config.targetFrameTimeMs;

    // P95 frame time (for detecting spikes)
    const sorted = [...this.frameTimes].sort((a, b) => a - b);
    const p95 = sorted[Math.floor(sorted.length * 0.95)];

    if (avg > target || p95 > this.config.criticalFrameTimeMs) {
      // Over budget
      this.consecutiveBadWindows++;
      this.consecutiveGoodWindows = 0;

      if (this.consecutiveBadWindows >= this.config.downgradeThreshold) {
        this.downgrade(`avg ${avg.toFixed(1)}ms > target ${target}ms (p95: ${p95.toFixed(1)}ms)`);
        this.consecutiveBadWindows = 0;
      }
    } else if (avg < target * 0.6) {
      // Well under budget — could handle more quality
      this.consecutiveGoodWindows++;
      this.consecutiveBadWindows = 0;

      if (this.consecutiveGoodWindows >= this.config.upgradeThreshold) {
        this.upgrade(`avg ${avg.toFixed(1)}ms << target ${target}ms`);
        this.consecutiveGoodWindows = 0;
      }
    } else {
      // Within budget — stable
      this.consecutiveGoodWindows = 0;
      this.consecutiveBadWindows = 0;
    }
  }

  private downgrade(reason: string): void {
    if (this.currentTier >= QUALITY_TIERS.length - 1) {
      this.log(`Already at minimum quality, cannot downgrade further`);
      return;
    }

    this.currentTier++;
    this.lastAdjustmentTime = performance.now();
    this.applyCurrentLevel(`downgrade: ${reason}`);
  }

  private upgrade(reason: string): void {
    if (this.currentTier <= 0) return;

    this.currentTier--;
    this.lastAdjustmentTime = performance.now();
    // Reset good windows counter — need to prove stability at new level
    this.consecutiveGoodWindows = 0;
    this.applyCurrentLevel(`upgrade: ${reason}`);
  }

  private applyCurrentLevel(reason: string): void {
    const level = this.getCurrentLevel();
    this.log(`Quality → ${level.label} (tier ${level.tier}): ${reason}`);

    for (const cb of this.applyCallbacks) {
      cb(level);
    }

    this.config.onQualityChange(level, reason);
  }

  private getAverageFrameTime(): number {
    const count = Math.min(this.totalFrames, this.config.windowSize);
    if (count === 0) return 0;
    let sum = 0;
    for (let i = 0; i < count; i++) {
      sum += this.frameTimes[i];
    }
    return sum / count;
  }

  private log(msg: string): void {
    if (this.config.debug) {
      console.log(`[AdaptiveQuality] ${msg}`);
    }
  }
}
