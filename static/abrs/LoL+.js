class LoLPlusABR {
  constructor() {
    this.minProfile = 0;
    this.maxProfile = 3;
    this.profile = 2;
    this.latencyTargetMs = 120;
    this.width = 800;
    this.height = 600;
    this.win = 30;
    this.logThr = [];
    this.serverMsByRes = new Map();
    this.lastBytes = 120_000;
    this.lastProfile = this.profile;
    this.lastPixels = this.width * this.height;
    this._t0 = null;
    this.wQual = 1.0;
    this.wRisk = 2.0;
    this.wSwitch = 0.15;
    this.upCooldownMs = 400;
    this.downCooldownMs = 300;
    this._lastChangeAt = 0;
    this.eps = 0.15;
    this.epsDecay = 0.97;
    this.clicks = 0;
    this.lastThroughputBps = 0;
    this.debug = true;
  }

  setResolution(w, h) {
    if (w > 0 && h > 0) { this.width = w; this.height = h; }
  }

  pickProfile() {
    const now = performance.now();
    const pixels = this.width * this.height;
    const medServer = this._median(this._arrForRes(this.width, this.height)) || this._serverDefault(pixels);
    const budgetNet = Math.max(1, this.latencyTargetMs - medServer);
    const { mu, sigma } = this._fitLogThr();
    const K = this._bytesPerPixelCoef();
    const cands = [];
    for (let p = this.minProfile; p <= this.maxProfile; p++) {
      const Xp = Math.max(1024, Math.round(K * pixels / Math.pow(2, p)));
      const { risk, expectTotal } = this._riskAndExpect(Xp, mu, sigma, medServer, budgetNet);
      const qual = this._qualityReward(p);
      const switchCost = (p !== this.profile) ? 1 : 0;
      const score = this.wQual * qual - this.wRisk * risk - this.wSwitch * switchCost;
      cands.push({ p, Xp, risk, expectTotal, score });
    }
    const canUp = (now - this._lastChangeAt) >= this.upCooldownMs;
    if (Math.random() < this.eps && canUp) {
      const safeHigher = cands
        .filter(c => c.p < this.profile && c.risk < 0.25)
        .sort((a, b) => a.risk - b.risk);
      if (safeHigher.length) {
        this._setProfile(safeHigher[0].p);
        if (this.debug) console.log(`[LoL+Click] explore -> p=${safeHigher[0].p} risk=${safeHigher[0].risk.toFixed(2)}`);
      }
    } else {
      cands.sort((a, b) => b.score - a.score);
      const best = cands[0];
      const isUp = best.p < this.profile;
      const isDown = best.p > this.profile;
      const canDown = (now - this._lastChangeAt) >= this.downCooldownMs;
      if ((isUp && canUp) || (isDown && canDown)) {
        this._setProfile(best.p);
        if (this.debug) {
          console.log(`[LoL+Click] choose p=${best.p} `
            + `score=${best.score.toFixed(3)} risk=${best.risk.toFixed(2)} `
            + `expTotal=${best.expectTotal.toFixed(1)}ms`
          );
        }
      }
    }
    this.clicks++;
    if (this.clicks % 10 === 0) this.eps = Math.max(0.05, this.eps * this.epsDecay);
    return this.profile;
  }

  startRequest() { this._t0 = performance.now(); }

  endRequest(contentLengthBytes, _rx = 0, _ry = 0, renderMs = NaN) {
    if (this._t0 == null) return;
    const dt = performance.now() - this._t0;
    this._t0 = null;
    const hasServer = Number.isFinite(renderMs) && renderMs >= 0;
    const netMs = hasServer ? Math.max(1, dt - renderMs) : dt;
    if (hasServer) this._push(this._arrForRes(this.width, this.height), renderMs);
    if (contentLengthBytes > 0) {
      this.lastBytes = contentLengthBytes;
      this.lastProfile = this.profile;
      this.lastPixels = this.width * this.height;
      const thrBytesPerMs = contentLengthBytes / Math.max(1, netMs);
      const z = Math.log(Math.max(thrBytesPerMs, 1e-6));
      this._push(this.logThr, z);
    }
    if (contentLengthBytes > 0 && netMs > 0) {
      this.lastThroughputBps = (contentLengthBytes / 1000) / (netMs / 1000);
      console.log(`[ABR] throughput = ${this.lastThroughputBps.toFixed(1)} Kb/s`);
    }
    if (this.debug) {
      const inst_bps = contentLengthBytes > 0 ? (contentLengthBytes * 8) / (netMs / 1000) : 0;
      console.debug(`[LoL+Click] done total=${dt.toFixed(1)}ms `
        + (hasServer ? `srv=${renderMs.toFixed(1)}ms ` : ``)
        + `net=${netMs.toFixed(1)}ms size=${contentLengthBytes}B `
        + `thr≈${(inst_bps / 1000).toFixed(0)} kbps res=${this.width}x${this.height} p=${this.profile}`);
    }
  }

  _bytesPerPixelCoef() {
    if (this.lastPixels > 0) {
      return Math.max(0.1, (this.lastBytes * Math.pow(2, this.lastProfile)) / this.lastPixels);
    }
    return 0.2;
  }

  _arrForRes(w, h) {
    const key = `${w}x${h}`;
    if (!this.serverMsByRes.has(key)) this.serverMsByRes.set(key, []);
    return this.serverMsByRes.get(key);
  }

  _serverDefault(pixels) {
    const base = 40;
    const refPx = 800 * 600;
    return base + 20 * Math.min(2.0, pixels / refPx);
  }

  _riskAndExpect(Xp, mu, sigma, medServer, budgetNet) {
    const c = 0.70;
    const thr_q = Math.exp(mu - c * sigma);
    const expectedNetMs = Xp / Math.max(thr_q, 1e-6);
    const expectTotal = expectedNetMs + medServer;
    let risk = 1.0;
    const needNet = Math.max(1, this.latencyTargetMs - medServer);
    if (sigma > 1e-6) {
      const y = Math.log(Xp / needNet);
      const t = (y - mu) / sigma;
      risk = this._stdNormCdf(t);
    } else {
      risk = (Math.exp(mu) < (Xp / needNet)) ? 1.0 : 0.0;
    }
    return { risk: this._clamp(risk, 0, 1), expectTotal };
  }

  _fitLogThr() {
    const n = this.logThr.length;
    if (n === 0) return { mu: Math.log(100), sigma: 0.7 };
    const mu = this.logThr.reduce((a, b) => a + b, 0) / n;
    const var_ = this.logThr.reduce((a, b) => a + (b - mu) * (b - mu), 0) / Math.max(1, n - 1);
    return { mu, sigma: Math.sqrt(Math.max(var_, 1e-6)) };
  }

  _qualityReward(p) {
    return [1.0, 0.82, 0.60, 0.42][p] ?? 0.2;
  }

  _setProfile(p) { if (p !== this.profile) { this.profile = p; this._lastChangeAt = performance.now(); } }

  _push(arr, v) { arr.push(v); if (arr.length > this.win) arr.shift(); }

  _median(arr) { const a = arr.slice().sort((x, y) => x - y); const m = a.length >> 1; return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2; }

  _clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }

  _stdNormCdf(t) {
    const b1 = 0.31938153, b2 = -0.356563782, b3 = 1.781477937, b4 = -1.821255978, b5 = 1.330274429, p = 0.2316419, c = 0.39894228;
    if (t >= 0) { const k = 1 / (1 + p * t); return 1 - c * Math.exp(-t * t / 2) * k * (b1 + k * (b2 + k * (b3 + k * (b4 + k * b5)))); }
    return 1 - this._stdNormCdf(-t);
  }
}
