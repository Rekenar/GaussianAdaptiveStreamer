// /static/abr_l2a.js
class L2A_ABR {
  /**
   * L2A-LL style ABR for request/response image rendering.
   * Online-learning (multiplicative weights / EXP3-like) over discrete profiles.
   * Now server-aware: consumes server render time to isolate network+browser latency.
   */
  constructor(_resolutionSelect) {
    // Profiles: 0=best quality, 3=lowest
    this.minProfile = 0;
    this.maxProfile = 3;
    this.profile = 3;                 // conservative start
    this.scaleLadder = [1.0, 0.75, 0.5, 0.35];

    // Online-learning state
    const K = this.maxProfile - this.minProfile + 1;
    this.K = K;
    this.weights = Array(K).fill(1);  // multiplicative weights
    this.gamma = 0.05;                // exploration
    this.eta = 0.25;                  // learning rate

    // Runtime stats (rolling)
    this.windowSize   = 12;
    this.latTotal     = [];           // total wall time (ms)
    this.latServer    = [];           // server render time (ms)
    this.latNet       = [];           // client/network time = total - server (ms)
    this.thrBytesPerMsNet = [];       // bytes / netMs (throughput estimate)
    this.lastBytes    = 0;

    // QoE / loss shaping
    this.latencyTargetMs = 120;       // end-to-end target
    this.aOvershoot = 2.0;            // weight for overshoot (stall risk)     [total]
    this.bLatency   = 0.5;            // gentle pressure on absolute latency   [total]
    this.bNet       = 0.35;           // gentle pressure on net time           [net]
    this.sSwitch    = 0.4;            // smoothness (penalize switches)
    this.qQuality   = 0.8;            // prefer visual quality

    // Stability
    this.cooldownMs = 350;
    this.lastChangeAt = 0;
    this.upgradeMargin = 0.1;         // require margin to upgrade quality

    // Misc
    this._t0 = null;
    this.prevProfile = undefined;
    this.debug = false;
  }

  pickProfile() {
    // Convert weights -> probabilities
    const probs = this._probsFromWeights();
    const best = this._argmax(probs);

    // Cooldown & hysteresis (avoid rapid flips)
    const now = performance.now();
    const canChange = (now - this.lastChangeAt) >= this.cooldownMs;
    if (!canChange && best !== this.profile) return this.profile;

    // Only upgrade quality if clearly better
    if (best < this.profile) {
      if (probs[best] - probs[this.profile] < this.upgradeMargin) return this.profile;
    }

    this.profile = best;
    this.lastChangeAt = now;
    return this.profile;
  }

  startRequest() { this._t0 = performance.now(); }

  /**
   * @param {number} contentLengthBytes
   * @param {number} _rx (unused)
   * @param {number} _ry (unused)
   * @param {number} renderMs OPTIONAL server render time (from X-Render-Time-Ms)
   */
  endRequest(contentLengthBytes = 0, _rx = 0, _ry = 0, renderMs = NaN) {
    if (this._t0 == null) return;
    const dtTotal = performance.now() - this._t0; // observed end-to-end time (ms)
    this._t0 = null;

    const hasServer = Number.isFinite(renderMs) && renderMs >= 0;
    const dtNet = hasServer ? Math.max(1, dtTotal - renderMs) : dtTotal;

    if (contentLengthBytes > 0) this.lastBytes = contentLengthBytes;

    // Update rolling stats
    this._push(this.latTotal, dtTotal);
    if (hasServer) {
      this._push(this.latServer, renderMs);
      this._push(this.latNet, dtNet);
      if (contentLengthBytes > 0) this._push(this.thrBytesPerMsNet, contentLengthBytes / dtNet);
    } else {
      if (contentLengthBytes > 0) this._push(this.thrBytesPerMsNet, contentLengthBytes / dtTotal);
    }

    // Compute realized loss (bandit feedback) for the chosen action only
    const lossChosen = this._lossFromObservation(dtTotal, dtNet, this.profile);

    // EXP3-style update
    const probs = this._probsFromWeights();
    const idx = this.profileIndex(this.profile);
    const pChosen = Math.max(1e-6, probs[idx]);
    const unbiasedLoss = lossChosen / pChosen;
    this.weights[idx] *= Math.exp(-this.eta * unbiasedLoss);

    // Cap weights to avoid numeric blow-up
    const cap = 1e6;
    for (let i = 0; i < this.K; i++) this.weights[i] = Math.min(this.weights[i], cap);

    if (this.debug) {
      if (hasServer) {
        const thr = this.thrBytesPerMsNet.length ? this._median(this.thrBytesPerMsNet) : 0;
        console.log(`[L2A] total=${dtTotal.toFixed(1)}ms server=${renderMs.toFixed(1)}ms `
          + `net=${dtNet.toFixed(1)}ms thr=${thr.toFixed(3)} B/ms `
          + `loss=${lossChosen.toFixed(3)} w=${this.weights.map(w=>w.toFixed(2))}`);
      } else {
        console.log(`[L2A] total=${dtTotal.toFixed(1)}ms (server N/A) `
          + `loss=${lossChosen.toFixed(3)} w=${this.weights.map(w=>w.toFixed(2))}`);
      }
    }
  }

  // ---- Loss model (lower is better) ----------------------------------------
  _lossFromObservation(dtTotalMs, dtNetMs, usedProfile) {
    const t = this.latencyTargetMs;

    // End-to-end pressure: overshoot + absolute total latency
    const over = Math.max(0, dtTotalMs - t) / t;    // stall risk vs target
    const lat  = dtTotalMs / t;

    // Network-only pressure: prefer actions that reduce network+browser time
    const net  = dtNetMs / t;

    // Smoothness penalty
    const switchPenalty = (usedProfile !== this.prevProfile) ? 1 : 0;

    // Quality penalty (higher when quality is low)
    const qual = 1 - this._qualityReward(usedProfile);

    // Remember for next round
    this.prevProfile = usedProfile;

    return this.aOvershoot * over
         + this.bLatency   * lat
         + this.bNet       * net
         + this.sSwitch    * switchPenalty
         + this.qQuality   * qual;
  }

  _qualityReward(p) {
    // Map profile -> [0..1] reward; tie to VMAF/PSNR buckets if available.
    const rewards = [1.0, 0.82, 0.6, 0.4];
    return rewards[p] ?? 0.3;
  }

  // ---- Probabilities & helpers ---------------------------------------------
  _probsFromWeights() {
    // EXP3-style mixing with exploration gamma
    const sumW = this.weights.reduce((a,b)=>a+b, 0) || 1;
    const base = this.weights.map(w => w / sumW);
    const eps = this.gamma / this.K;
    return base.map(p => (1 - this.gamma) * p + eps);
  }

  profileIndex(p) { return p - this.minProfile; }

  _argmax(arr) {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i; }
    return this.minProfile + bi;
  }

  _push(a, v) { a.push(v); if (a.length > this.windowSize) a.shift(); }

  _median(arr) {
    const a = arr.slice().sort((x,y)=>x-y);
    const m = Math.floor(a.length/2);
    return a.length % 2 ? a[m] : (a[m-1] + a[m]) / 2;
  }
}
