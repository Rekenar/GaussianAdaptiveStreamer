// /static/abr_l2a.js
class L2A_ABR {
  constructor(_resolutionSelect) {
    this.minProfile = 0;
    this.maxProfile = 3;
    this.profile = 3;
    this.scaleLadder = [1.0, 0.75, 0.5, 0.35];

    this.width = 800;
    this.height = 600;

    this.K = this.maxProfile - this.minProfile + 1;
    this.logW = Array(this.K).fill(0.0);

    this.gamma = 0.01; 
    this.eta = 0.01;   

    this.windowSize = 30;
    this.latTotal = [];  
    this.latServer = []; 
    this.latNet = [];     
    this.thrBytesPerMsNet = [];

    this.latencyTargetMs = 140; 
    this.minNetBudgetMs = 20;  

    this.aOvershoot = 2.0;
    this.bLatency = 0.4; 
    this.sSwitch = 1;
    this.qQuality = 0.55;

    this.cooldownMs = 1500;
    this.lastChangeAt = 0;

    this.clicks = 0;
    this.gammaMin = 0.005;
    this.gammaDecay = 0.98; 

    this.lastBytes = 12000;  
    this.lastProfile = this.profile;
    this.lastPixels = this.width * this.height;

    this._t0 = null;
    this.prevProfile = undefined;
    this.debug = true;

    this._lastProbs = Array(this.K).fill(1 / this.K);
    this._lastPChosen = 1.0;
    this.lastThroughputBps = 0;
  }

  setResolution(w, h) {
    if (w > 0 && h > 0) {
      this.width = w; this.height = h;
    }
  }

  pickProfile() {
    const now = performance.now();
    const canChange = (now - this.lastChangeAt) >= this.cooldownMs;

    const medServer = this._median(this.latServer) || this._serverDefault();
    const netBudget = Math.max(this.minNetBudgetMs, this.latencyTargetMs - medServer);

    const { mu, sigma } = this._fitLogThr();
    const K = this._bytesPerPixelCoef();
    const pixels = this.width * this.height;

    let probs = this._probsFromLogWeights();
    this._lastProbs = probs.slice();

    const safety = [];
    const riskCap = 0.70; // Slightly more conservative
    const headroom = 1.10;    
    for (let p = this.minProfile; p <= this.maxProfile; p++) {
      const bytes = this._bytesFor(p, K, pixels);
      const { expectTotal, risk } = this._riskAndExpect(bytes, mu, sigma, medServer, netBudget);
      const ok = (risk <= riskCap) && (expectTotal <= headroom * this.latencyTargetMs);
      safety.push(ok ? 1 : 0);
    }

    if (safety.every(x => x === 0)) {
      let bestP = this.profile, bestR = +Infinity;
      for (let p = this.minProfile; p <= this.maxProfile; p++) {
        const r = this._riskForP(p, K, pixels, mu, sigma, medServer, netBudget);
        if (r < bestR) { bestR = r; bestP = p; }
      }
      const mask = Array(this.K).fill(0); mask[this.profileIndex(bestP)] = 1;
      probs = this._renorm(probs.map((p, i) => p * mask[i]));
    } else {
      probs = this._renorm(probs.map((p, i) => p * safety[i]));
    }

    if (!canChange) {
      this._lastPChosen = 1.0;
      return this.profile;
    }

    const candidate = this._pickByProbs(probs);
    this._lastPChosen = Math.max(1e-6, probs[this.profileIndex(candidate)]);

    if (candidate !== this.profile) {
      this.prevProfile = this.profile;
      this.profile = candidate;
      this.lastChangeAt = now;
    }

    this.clicks++;
    if (this.clicks % 12 === 0) {
      this.gamma = Math.max(this.gammaMin, this.gamma * this.gammaDecay);
    }
    return this.profile;
  }

  startRequest() { this._t0 = performance.now(); }

  endRequest(contentLengthBytes = 0, _rx = 0, _ry = 0, renderMs = NaN) {
    if (this._t0 == null) return;

    const dtTotal = performance.now() - this._t0;
    this._t0 = null;

    const hasServer = Number.isFinite(renderMs) && renderMs >= 0;
    const netMs = hasServer ? Math.max(1, dtTotal - renderMs) : dtTotal;

    this._push(this.latTotal, dtTotal);
    if (hasServer) this._push(this.latServer, renderMs);
    this._push(this.latNet, netMs);

    if (contentLengthBytes > 0) {
      this.lastBytes = contentLengthBytes;
      this.lastProfile = this.profile;
      this.lastPixels = this.width * this.height;
      this._push(this.thrBytesPerMsNet, contentLengthBytes / netMs);
    }

    if (contentLengthBytes > 0 && netMs > 0) {
      this.lastThroughputBps = (contentLengthBytes / 1000) / (netMs / 1000);
    }

    const medServer = this._median(this.latServer) || this._serverDefault();
    const netBudget = Math.max(this.minNetBudgetMs, this.latencyTargetMs - medServer);

    const usedProfile = this.profile;
    const lossChosen = this._lossFromNetObservation(netMs, usedProfile, netBudget);

    const pChosen = Math.max(1e-6, this._lastPChosen);
    const unbiasedLoss = lossChosen / pChosen;
    const idx = this.profileIndex(usedProfile);
    this.logW[idx] += (-this.eta * unbiasedLoss);

    const maxLogW = Math.max(...this.logW);
    for (let i = 0; i < this.K; i++) this.logW[i] -= maxLogW;

    if (this.debug) {
        console.log(`[L2A] net=${netMs.toFixed(1)}ms loss=${lossChosen.toFixed(3)} p=${pChosen.toFixed(3)} prof=${usedProfile}`);
    }
  }

  _lossFromNetObservation(netMs, usedProfile, netBudgetMs) {
    const t = Math.max(1, netBudgetMs);
    const overC = Math.max(0, netMs - t) / t; 
    const latC = netMs / this.latencyTargetMs; 
    const switchPenalty = (usedProfile !== this.prevProfile) ? 1 : 0;
    const qual = 1 - this._qualityReward(usedProfile);

    let L = (this.aOvershoot * overC) + (this.bLatency * latC) + (this.sSwitch * switchPenalty) + (this.qQuality * qual);
    
    this.prevProfile = usedProfile;
    return L;
  }

_fitLogThr() {
  const arr = this.thrBytesPerMsNet;
  if (arr.length < 10) return { mu: Math.log(100), sigma: 0.5 }; 
  
  const z = arr.map(x => Math.log(Math.max(x, 1e-6)));
  const n = z.length;
  const mu = z.reduce((a, b) => a + b, 0) / n;

  const rawVar = z.reduce((a, b) => a + (b - mu) * (b - mu), 0) / Math.max(1, n - 1);
  const sigma = Math.sqrt(Math.max(rawVar, 0.1));
  
  return { mu, sigma: Math.min(sigma, 0.4) };
}

  _riskAndExpect(bytes, mu, sigma, medServer, netBudget) {
    const thr_q = Math.exp(mu - 0.67 * sigma); 
    const expectedNetMs = bytes / Math.max(thr_q, 1e-6);
    const expectTotal = expectedNetMs + medServer;

    let risk = 1.0;
    const needNet = Math.max(1, this.latencyTargetMs - medServer);
    if (sigma > 1e-6) {
      const y = Math.log(bytes / needNet);
      const t = (y - mu) / sigma;
      risk = this._stdNormCdf(t);
    }
    return { expectTotal, risk: this._clamp(risk, 0, 1) };
  }

_probsFromLogWeights() {
    const temperature = 0.2;
    const maxL = Math.max(...this.logW);
    
    const w = this.logW.map(L => Math.exp((L - maxL) / temperature));
    
    const sumW = w.reduce((a, b) => a + b, 0) || 1;
    const base = w.map(x => x / sumW);

    const eps = this.gamma / this.K;
    const mixed = base.map(p => (1 - this.gamma) * p + eps);
    return this._renorm(mixed);
}

  _percentile(arr, p) {
    if (!arr.length) return 0;
    const sorted = arr.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length * p)];
  }

  _median(arr) { return this._percentile(arr, 0.5); }

  _bytesFor(p, K, pixels) {
    return Math.max(1024, Math.round(K * pixels / Math.pow(2, p)));
  }

  _bytesPerPixelCoef() {
    const px = this.lastPixels || (this.width * this.height);
    if (px > 0) return Math.max(0.05, (this.lastBytes * Math.pow(2, this.lastProfile)) / px);
    return 0.2;
  }

  _serverDefault() {
    const base = 40, refPx = 800 * 600, px = this.width * this.height;
    return base + 20 * Math.min(2.0, px / refPx);
  }

  _riskForP(p, K, pixels, mu, sigma, medServer, netBudget) {
    const bytes = this._bytesFor(p, K, pixels);
    return this._riskAndExpect(bytes, mu, sigma, medServer, netBudget).risk;
  }

  _qualityReward(p) { return 1 / (1 + 0.6 * p); }

  _pickByProbs(probs) {
    const r = Math.random(); let c = 0;
    for (let i = 0; i < probs.length; i++) { c += probs[i]; if (r <= c) return this.minProfile + i; }
    return this.maxProfile;
  }

  _renorm(arr) {
    const s = arr.reduce((a, b) => a + b, 0) || 1;
    return arr.map(x => x / s);
  }

  profileIndex(p) { return p - this.minProfile; }
  _push(a, v) { a.push(v); if (a.length > this.windowSize) a.shift(); }
  _clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }

  _stdNormCdf(t) {
    const b1 = 0.31938153, b2 = -0.356563782, b3 = 1.781477937, b4 = -1.821255978, b5 = 1.330274429, p = 0.2316419, c = 0.39894228;
    if (t >= 0) { const k = 1 / (1 + p * t); return 1 - c * Math.exp(-t * t / 2) * k * (b1 + k * (b2 + k * (b3 + k * (b4 + k * b5)))); }
    return 1 - this._stdNormCdf(-t);
  }
}