// /static/abr.js
class ABRAlgorithm {
  /**
   * Extremely simple ABR returning profile 0..3.
   * Uses only recent request latency (ms) with a rolling window + hysteresis.
   * Never changes width/height.
   */
  constructor(_resolutionSelect) {
    this.profile = 0;
    this.minProfile = 0;
    this.maxProfile = 3;

    // Rolling timings (ms)
    this.window = [];
    this.windowSize = 5;

    // Hysteresis thresholds on median latency
    this.upgradeMs = 95;    // faster than -> try higher quality (lower profile)
    this.downgradeMs = 150; // slower than -> reduce quality (higher profile)

    // Require stability before changing
    this.needGoodCount = 2; // consecutive “good” decisions before upgrade
    this.goodStreak = 0;

    // Cooldown to avoid flapping
    this.lastChangeAt = 0;
    this.changeCooldownMs = 400;

    // Timing per request
    this._t0 = null;

    // Optional logs
    this.debug = false;

    // Small nudge so it doesn’t stick at 0 on pristine networks
    this.startAt = 1;        // set to 0 if you prefer truly neutral start
    this._initialized = false;
  }

  pickProfile() {
    if (!this._initialized) {
      this.profile = Math.min(this.maxProfile, Math.max(this.minProfile, this.startAt));
      this._initialized = true;
      if (this.debug) console.log(`[ABR] init profile=${this.profile}`);
    }
    return this.profile;
  }

  startRequest() {
    this._t0 = performance.now();
  }

  endRequest(_contentLengthBytes) {
    if (this._t0 == null) return;
    const dt = performance.now() - this._t0;
    this._t0 = null;

    // Keep a bounded window
    this.window.push(dt);
    if (this.window.length > this.windowSize) this.window.shift();

    // Not enough data yet? bail.
    if (this.window.length < Math.min(3, this.windowSize)) {
      if (this.debug) console.log(`[ABR] warmup dt=${dt.toFixed(1)}ms`);
      return;
    }

    // Median latency
    const med = this._median(this.window);

    if (this.debug) {
      console.log(`[ABR] dt=${dt.toFixed(1)}ms med=${med.toFixed(1)}ms prof=${this.profile}`);
    }

    const now = performance.now();
    const canChange = (now - this.lastChangeAt) >= this.changeCooldownMs;

    if (med > this.downgradeMs) {
      this.goodStreak = 0;
      if (canChange) this._bump(+1, "slow");
    } else if (med < this.upgradeMs) {
      this.goodStreak++;
      if (this.goodStreak >= this.needGoodCount && canChange) {
        this._bump(-1, "fast");
        this.goodStreak = 0;
      }
    } else {
      // In the middle band, reset streak
      this.goodStreak = 0;
    }
  }

  _bump(delta, why) {
    const old = this.profile;
    this.profile = Math.max(this.minProfile, Math.min(this.maxProfile, this.profile + delta));
    this.lastChangeAt = performance.now();
    if (this.debug && old !== this.profile) {
      console.log(`[ABR] profile ${old} -> ${this.profile} (${why})`);
    }
  }

  _median(arr) {
    const a = arr.slice().sort((x, y) => x - y);
    const mid = Math.floor(a.length / 2);
    return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
  }
}

window.ABRAlgorithm = ABRAlgorithm;
