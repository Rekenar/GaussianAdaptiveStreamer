// /static/abr_lolp.js
class LoLPlusABR {
  /** LoL+-style ABR for request/response image rendering.
   * QoE score = wQ*qual - wR*risk - wS*switch - wL*latGap
   * - Uses server render time (if provided) to separate network+browser vs server.
   * - Throughput is estimated from bytes / netMs (total - server).
   * - Risk compares expected TOTAL time (expectedNet + medianServer) to target.
   */
  constructor(_resolutionSelect) {
    this.minProfile = 0;
    this.maxProfile = 3;
    this.profile = 3;
    this.scaleLadder = [1.0, 0.75, 0.5, 0.35];

    // Rolling stats
    this.windowSize = 12;
    this.latTotal = [];        // total wall time (ms)
    this.latNet   = [];        // total - server (ms)
    this.latServer = [];       // server render (ms)
    this.thrBytesPerMsNet = []; // bytes / netMs
    this.lastContentLen = 0;

    // Targets / thresholds
    this.latencyTargetMs = 120; // end-to-end target
    this.cooldownMs = 350;
    this.lastChangeAt = 0;

    // QoE weights
    this.wQ = 1.0;
    this.wR = 2.2;
    this.wS = 0.6;
    this.wL = 0.8;

    // Online tiny adaptation
    this.learnRate = 0.02;

    // Misc
    this._t0 = null;
    this.debug = true;
  }

  pickProfile() {
    return this.profile;
  }

  startRequest() { this._t0 = performance.now(); }

  /**
   * @param {number} contentLengthBytes
   * @param {number} _rx (unused)
   * @param {number} _ry (unused)
   * @param {number} renderMs OPTIONAL server render time from header
   */
  endRequest(contentLengthBytes, _rx = 0, _ry = 0, renderMs = NaN) {
    if (this._t0 == null) return;
    const dt = performance.now() - this._t0; // total
    this._t0 = null;

    const hasServer = Number.isFinite(renderMs) && renderMs >= 0;
    const netMs = hasServer ? Math.max(1, dt - renderMs) : dt;

    this.lastContentLen = contentLengthBytes || this.lastContentLen;

    // Push bounded windows
    this._push(this.latTotal, dt);
    if (hasServer) {
      this._push(this.latServer, renderMs);
      this._push(this.latNet, netMs);
      if (contentLengthBytes) this._push(this.thrBytesPerMsNet, contentLengthBytes / netMs);
    } else {
      if (contentLengthBytes) this._push(this.thrBytesPerMsNet, contentLengthBytes / dt);
    }

    // Need at least a few samples
    const enough = this.latTotal.length >= Math.min(3, this.windowSize);
    if (!enough) return;

    // Medians
    const medTotal = this._median(this.latTotal);
    const medServer = this.latServer.length ? this._median(this.latServer) : 0;
    const medNet = this.latNet.length ? this._median(this.latNet) : Math.max(1, medTotal - medServer);

    // Throughput (bytes/ms) from net time
    const estThr = this.thrBytesPerMsNet.length
      ? this._median(this.thrBytesPerMsNet)
      : (this.lastContentLen > 0 ? this.lastContentLen / Math.max(1, medNet) : 0.001);

    // Score profiles
    const now = performance.now();
    const canChange = (now - this.lastChangeAt) >= this.cooldownMs;

    const current = this.profile;
    let best = current;
    let bestScore = -Infinity;

    for (let p = this.minProfile; p <= this.maxProfile; p++) {
      const score = this._scoreOf(p, estThr, medNet, medServer, medTotal);
      if (score > bestScore) { bestScore = score; best = p; }
    }

    // Hysteresis: require margin to upgrade quality; quick to downgrade
    const margin = 0.05;
    const currentScore = this._scoreOf(current, estThr, medNet, medServer, medTotal);
    const shouldUpgrade = (best < current) && ((bestScore - currentScore) > margin);
    const shouldDowngrade = (best > current);

    if (canChange && (shouldUpgrade || shouldDowngrade)) {
      if (this.debug) {
        console.log(`[LoL+] prof ${current} -> ${best} `
          + `(Δscore=${(bestScore - currentScore).toFixed(3)}, `
          + `medTotal=${medTotal.toFixed(1)}ms, medServer=${medServer.toFixed(1)}ms, medNet=${medNet.toFixed(1)}ms)`);
      }
      this.profile = best;
      this.lastChangeAt = performance.now();

      // Tiny “plus”: adapt weights
      if (medTotal > this.latencyTargetMs) this.wL += this.learnRate; // value latency more
      else this.wQ += this.learnRate * 0.5;                           // value quality slightly more

      // Clamp
      this.wQ = this._clamp(this.wQ, 0.5, 2.5);
      this.wR = this._clamp(this.wR, 1.0, 3.5);
      this.wS = this._clamp(this.wS, 0.2, 1.5);
      this.wL = this._clamp(this.wL, 0.3, 2.0);
    }

    if (this.debug) {
      if (hasServer) {
        console.debug(`[LoL+] total=${dt.toFixed(1)}ms server=${renderMs.toFixed(1)}ms net=${netMs.toFixed(1)}ms `
          + `thr=${estThr.toFixed(3)} B/ms prof=${this.profile}`);
      } else {
        console.debug(`[LoL+] total=${dt.toFixed(1)}ms thr=${estThr.toFixed(3)} B/ms prof=${this.profile}`);
      }
    }
  }

  // ---- QoE scoring ---------------------------------------------------------

  _scoreOf(p, estThrBytesPerMs, medNetMs, medServerMs, medTotalMs) {
    const scale = this.scaleLadder[p];
    const expectedBytes = Math.max(1, this.lastContentLen) * (scale ** 2);

    // Expected net transfer/processing time (client side)
    const expectedNetMs = expectedBytes / Math.max(estThrBytesPerMs, 1e-6);

    // Expected TOTAL time ≈ expectedNet + typical server render
    const expectedTotalMs = expectedNetMs + medServerMs;

    const quality = this._qualityReward(p);

    // Stall risk vs end-to-end target
    const stallRisk = Math.max(0, (expectedTotalMs - this.latencyTargetMs)) / this.latencyTargetMs;

    // Gap from target using observed total median (smoother than single-sample)
    const latencyGap = Math.max(0, medTotalMs - this.latencyTargetMs) / this.latencyTargetMs;

    const switchPenalty = (p !== this.profile) ? 1 : 0;

    return this.wQ * quality
         - this.wR * stallRisk
         - this.wS * switchPenalty
         - this.wL * latencyGap;
  }

  _qualityReward(p) {
    // Map profile index to quality reward
    const rewards = [1.0, 0.8, 0.55, 0.35];
    return rewards[p] ?? 0.2;
  }

  // ---- utils ---------------------------------------------------------------

  _push(arr, v) { arr.push(v); if (arr.length > this.windowSize) arr.shift(); }
  _median(arr) { const a = arr.slice().sort((x,y)=>x-y); const m = Math.floor(a.length/2); return a.length % 2 ? a[m] : (a[m-1]+a[m])/2; }
  _clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }
}
