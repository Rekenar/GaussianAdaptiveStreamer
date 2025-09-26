// /static/abr.js
class LatencyABR {
  /** Simple ABR returning profile 0..3.
   * Uses recent request latency with rolling window + hysteresis.
   * If server render time is available (X-Render-Time-Ms), it subtracts it
   * to focus on network+browser time.
   */
  constructor(_resolutionSelect) {
    this.profile = 0;
    this.minProfile = 0;
    this.maxProfile = 3;

    // Rolling windows
    this.windowTotal = [];  // total wall time per req (ms)
    this.windowNet   = [];  // total - serverRender (ms) when available
    this.windowSize = 12;

    // Thresholds
    this.upgradeMs = 95;    // faster than -> try higher quality (lower profile)
    this.downgradeMs = 150; // slower than -> reduce quality (higher profile)

    this.needGoodCount = 3; // consecutive “good” decisions before upgrade
    this.goodStreak = 0;    // cooldown logic
    this.lastChangeAt = 0;
    this.changeCooldownMs = 400;

    // Timing per request
    this._t0 = null;

    // Optional logs
    this.debug = true;

    // Start high-compression (lowest quality) to be safe
    this.startAt = 3;
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

  /**
   * @param {number} _contentLengthBytes - optional, for future use
   * @param {number} _rx - received image width (optional)
   * @param {number} _ry - received image height (optional)
   * @param {number} renderMs - OPTIONAL server render time in ms (from header)
   */
  endRequest(_contentLengthBytes = 0, _rx = 0, _ry = 0, renderMs = NaN) {
    if (this._t0 == null) return;

    const t1 = performance.now();
    const dt = t1 - this._t0;        // total observed latency on client
    this._t0 = null;

    // If server render time was sent, isolate network+browser time
    const hasServer = Number.isFinite(renderMs) && renderMs >= 0;
    const netMs = hasServer ? Math.max(1, dt - renderMs) : dt;

    // Keep bounded windows
    this.windowTotal.push(dt);
    if (this.windowTotal.length > this.windowSize) this.windowTotal.shift();

    if (hasServer) {
      this.windowNet.push(netMs);
      if (this.windowNet.length > this.windowSize) this.windowNet.shift();
    }

    // Not enough samples? bail
    const effectiveWindow = hasServer ? this.windowNet : this.windowTotal;
    if (effectiveWindow.length < Math.min(3, this.windowSize)) {
      if (this.debug) {
        console.log(`[ABR] warmup total=${dt.toFixed(1)}ms`
          + (hasServer ? ` server=${renderMs.toFixed(1)}ms net=${netMs.toFixed(1)}ms` : ''));
      }
      return;
    }

    const med = this._median(effectiveWindow);

    if (this.debug) {
      if (hasServer) {
        console.log(
          `[ABR] total=${dt.toFixed(1)}ms server=${renderMs.toFixed(1)}ms `
          + `net=${netMs.toFixed(1)}ms med(net)=${med.toFixed(1)}ms prof=${this.profile}`
        );
      } else {
        console.log(
          `[ABR] total=${dt.toFixed(1)}ms med(total)=${med.toFixed(1)}ms prof=${this.profile}`
        );
      }
    }

    // Hysteresis and cooldown
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
