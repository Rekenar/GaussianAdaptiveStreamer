// /static/abr_lolp_click.js
// LoL+-style for click-to-render: resolution affects server time, profile = compression (2^profile)

class LoLPlusABR {
  constructor() {
    // Profiles: index increases => more compression => fewer bytes, lower quality
    this.minProfile = 0;
    this.maxProfile = 3;
    this.profile = 2; // start mid
    this.latencyTargetMs = 120;

    // Resolution (set from UI)
    this.width = 800;
    this.height = 600;

    // Rolling stats
    this.win = 30;
    this.logThr = [];         // ln(bytes/ms), network only
    this.serverMsByRes = new Map(); // key "WxH" -> array of ms
    this.lastBytes = 120_000; // last JPEG size (bytes)
    this.lastProfile = this.profile;
    this.lastPixels = this.width * this.height;
    this._t0 = null;

    // Scoring weights
    this.wQual = 1.0;
    this.wRisk = 2.0;
    this.wSwitch = 0.15;

    // Hysteresis
    this.upCooldownMs = 400;
    this.downCooldownMs = 300;
    this._lastChangeAt = 0;

    // Exploration (ε-greedy to higher quality if safe)
    this.eps = 0.15;
    this.epsDecay = 0.97;
    this.clicks = 0;

    this.debug = true;
  }

  setResolution(w, h) {
    if (w > 0 && h > 0) { this.width = w; this.height = h; }
  }

  pickProfile() {
    // Decide at click-time using current stats and current resolution
    const now = performance.now();
    const pixels = this.width * this.height;

    const medServer = this._median(this._arrForRes(this.width, this.height)) || this._serverDefault(pixels);
    const budgetNet = Math.max(1, this.latencyTargetMs - medServer);

    const { mu, sigma } = this._fitLogThr();
    const K = this._bytesPerPixelCoef(); // learned from last sample

    // Build candidates
    const cands = [];
    for (let p = this.minProfile; p <= this.maxProfile; p++) {
      const Xp = Math.max(1024, Math.round(K * pixels / Math.pow(2, p))); // bytes at profile p
      const { risk, expectTotal } = this._riskAndExpect(Xp, mu, sigma, medServer, budgetNet);
      const qual = this._qualityReward(p);
      const switchCost = (p !== this.profile) ? 1 : 0;
      const score = this.wQual * qual - this.wRisk * risk - this.wSwitch * switchCost;
      cands.push({ p, Xp, risk, expectTotal, score });
    }

    // ε-greedy exploration to higher quality if not obviously risky
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

    // decay ε every 10 clicks
    this.clicks++;
    if (this.clicks % 10 === 0) this.eps = Math.max(0.05, this.eps * this.epsDecay);

    return this.profile;
  }

  startRequest() { this._t0 = performance.now(); }

  // contentLengthBytes: actual encoded bytes returned for THIS profile & resolution
  // renderMs: server render time provided by server for this image (if available)
  endRequest(contentLengthBytes, _rx=0, _ry=0, renderMs=NaN) {
    if (this._t0 == null) return;
    const dt = performance.now() - this._t0; // total
    this._t0 = null;

    const hasServer = Number.isFinite(renderMs) && renderMs >= 0;
    const netMs = hasServer ? Math.max(1, dt - renderMs) : dt;

    // Update server-time bucket for current resolution
    if (hasServer) this._push(this._arrForRes(this.width, this.height), renderMs);

    // Update last sample and throughput
    if (contentLengthBytes > 0) {
      this.lastBytes = contentLengthBytes;
      this.lastProfile = this.profile;
      this.lastPixels = this.width * this.height;

      const thrBytesPerMs = contentLengthBytes / Math.max(1, netMs);
      const z = Math.log(Math.max(thrBytesPerMs, 1e-6));
      this._push(this.logThr, z);
    }

    if (this.debug) {
      const inst_bps = contentLengthBytes > 0 ? (contentLengthBytes * 8) / (netMs / 1000) : 0;
      console.debug(`[LoL+Click] done total=${dt.toFixed(1)}ms `
        + (hasServer ? `srv=${renderMs.toFixed(1)}ms ` : ``)
        + `net=${netMs.toFixed(1)}ms size=${contentLengthBytes}B `
        + `thr≈${(inst_bps/1000).toFixed(0)} kbps res=${this.width}x${this.height} p=${this.profile}`);
    }
  }

  // ---------- internals ----------

  _bytesPerPixelCoef() {
    // Learn K ≈ bytes_per_pixel_at_profile0; use last sample to calibrate profiles
    // Model: bytes ≈ K * pixels / 2^profile  => K ≈ bytes * 2^profile / pixels
    if (this.lastPixels > 0) {
      return Math.max(0.1, (this.lastBytes * Math.pow(2, this.lastProfile)) / this.lastPixels);
    }
    // fallback
    return 0.2; // bytes per pixel at profile 0 (tunable)
  }

  _arrForRes(w, h) {
    const key = `${w}x${h}`;
    if (!this.serverMsByRes.has(key)) this.serverMsByRes.set(key, []);
    return this.serverMsByRes.get(key);
  }

  _serverDefault(pixels) {
    // Default 40–60 ms range mentioned; scale mildly with pixels
    // 40 ms baseline + 20 ms * relative area to 800x600
    const base = 40;
    const refPx = 800 * 600;
    return base + 20 * Math.min(2.0, pixels / refPx); // cap growth a bit
  }

  _riskAndExpect(Xp, mu, sigma, medServer, budgetNet) {
    // conservative quantile for expectation
    const c = 0.70; // ~24th percentile
    const thr_q = Math.exp(mu - c * sigma); // bytes/ms
    const expectedNetMs = Xp / Math.max(thr_q, 1e-6);
    const expectTotal = expectedNetMs + medServer;

    // Risk: P(total > target) ≈ P(thr < Xp / (target - medServer))
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
    if (n === 0) return { mu: Math.log(100), sigma: 0.7 }; // ≈100 B/ms default
    const mu = this.logThr.reduce((a,b)=>a+b,0)/n;
    const var_ = this.logThr.reduce((a,b)=>a+(b-mu)*(b-mu),0)/Math.max(1,n-1);
    return { mu, sigma: Math.sqrt(Math.max(var_, 1e-6)) };
  }

  _qualityReward(p) {
    // You can tune this vector to subjective quality vs compression
    return [1.0, 0.82, 0.60, 0.42][p] ?? 0.2;
  }

  _setProfile(p) { if (p !== this.profile) { this.profile = p; this._lastChangeAt = performance.now(); } }

  _push(arr, v) { arr.push(v); if (arr.length > this.win) arr.shift(); }

  _median(arr) { const a = arr.slice().sort((x,y)=>x-y); const m = a.length>>1; return a.length%2?a[m]:(a[m-1]+a[m])/2; }

  _clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }

  _stdNormCdf(t) {
    const b1=0.31938153,b2=-0.356563782,b3=1.781477937,b4=-1.821255978,b5=1.330274429,p=0.2316419,c=0.39894228;
    if (t >= 0) { const k=1/(1+p*t); return 1-c*Math.exp(-t*t/2)*k*(b1+k*(b2+k*(b3+k*(b4+k*b5)))); }
    return 1 - this._stdNormCdf(-t);
  }
}
