class LatencyABR {
  constructor(_resolutionSelect) {
    this.profile = 0;
    this.minProfile = 0;
    this.maxProfile = 3;

    // Rolling windows
    this.windowTotal = [];
    this.windowNet = [];
    this.windowSize = 30;          // ↑ for stability

    // Thresholds (deadband widened)
    this.upgradeMs = 32;          // was 50 (harder to upgrade)
    this.downgradeMs = 90;          // was 70 (easier to downgrade)

    // Hysteresis
    this.needGoodCount = 12;         // was 5
    this.goodStreak = 0;
    this.lastUpAt = 0;
    this.lastDownAt = 0;
    this.upCooldownMs = 3000;      // upgrades spaced out
    this.downCooldownMs = 600;       // downgrades remain responsive

    // Optional: track rough bytes-per-pixel to sanity-check upgrades
    this.bppWin = [];               // recent bytes-per-pixel (size / (rx*ry))
    this.bppWinSize = 20;


    this.lastThroughputBps = 0;

    this._t0 = null;
    this.debug = true;
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

  startRequest() { this._t0 = performance.now(); }

  endRequest(contentLengthBytes = 0, rx = 0, ry = 0, renderMs = NaN) {
    if (this._t0 == null) return;
    const t1 = performance.now();
    const dt = t1 - this._t0;
    this._t0 = null;

    const hasServer = Number.isFinite(renderMs) && renderMs >= 0;
    const netMs = hasServer ? Math.max(1, dt - renderMs) : dt;

    console.log("Content:" + contentLengthBytes);
    console.log("Net MS: " + netMs);

    if (contentLengthBytes > 0 && netMs > 0) {
      this.lastThroughputBps = (contentLengthBytes / 1000) / (netMs / 1000); // Bytes per second
      console.log(`[ABR] throughput = ${this.lastThroughputBps.toFixed(1)} B/s`);
    }

    // record windows
    this.windowTotal.push(dt); if (this.windowTotal.length > this.windowSize) this.windowTotal.shift();
    if (hasServer) { this.windowNet.push(netMs); if (this.windowNet.length > this.windowSize) this.windowNet.shift(); }

    // update bpp if we know size and dimensions
    if (contentLengthBytes > 0 && rx > 0 && ry > 0) {
      const bpp = contentLengthBytes / (rx * ry);
      this.bppWin.push(bpp);
      if (this.bppWin.length > this.bppWinSize) this.bppWin.shift();
    }

    const effective = hasServer ? this.windowNet : this.windowTotal;
    if (effective.length < Math.min(3, this.windowSize / 2)) {
      if (this.debug) console.log(`[ABR] warmup total=${dt.toFixed(1)}ms${hasServer ? ` server=${renderMs.toFixed(1)}ms net=${netMs.toFixed(1)}ms` : ''}`);
      return;
    }

    const med = this._median(effective);
    if (this.debug) {
      if (hasServer) {
        console.log(`[ABR] total=${dt.toFixed(1)}ms server=${renderMs.toFixed(1)}ms net=${netMs.toFixed(1)}ms med(net)=${med.toFixed(1)}ms prof=${this.profile}`);
      } else {
        console.log(`[ABR] total=${dt.toFixed(1)}ms med(total)=${med.toFixed(1)}ms prof=${this.profile}`);
      }
    }

    const now = performance.now();
    const canUp = (now - this.lastUpAt) >= this.upCooldownMs;
    const canDown = (now - this.lastDownAt) >= this.downCooldownMs;

    if (med > this.downgradeMs) {
      this.goodStreak = 0;
      if (canDown) this._bump(+1, "slow");
      return;
    }

    if (med < this.upgradeMs) {
      this.goodStreak++;
      if (this.goodStreak >= this.needGoodCount && canUp) {
        if (this._nextStepLooksSafe(med)) {
          this._bump(-1, "fast-safe");
          this.goodStreak = 0;
        } else {
          if (this.debug) console.log("[ABR] upgrade vetoed: next profile predicted to exceed budget");
        }
      }
      return;
    }

    // middle band
    this.goodStreak = 0;
  }

  _nextStepLooksSafe(currentMedNetMs) {
    const next = this.profile - 1;
    if (next < this.minProfile) return false;

    // Use median BPP as a rough predictor; fall back to conservative default
    const bpp = this.bppWin.length ? this._median(this.bppWin) : 0.0025; // bytes per pixel fallback
    // Assume resolution increases when upgrading quality; if you know exact ladder, plug it in.
    // If you know rx, ry for the *target* profile, estimate bytes:
    // Here we approximate by saying next profile increases bytes by ~1.6× vs current.
    const bytesFactor = 1.6; // tune to your ladder
    const lastSize = this._lastContentLength || 0; // populate this from caller if available
    const estBytes = lastSize > 0 ? lastSize * bytesFactor : (bpp * (this._lastRx || 1280) * (this._lastRy || 720)) * bytesFactor;

    // Estimate download time by scaling with current median net time and last size
    // If we don't have last size, use a simple proportional check vs threshold.
    const estNetMs = lastSize > 0 ? currentMedNetMs * (estBytes / lastSize) : currentMedNetMs * bytesFactor;

    // Only allow upgrade if predicted time remains below the *downgrade* budget with margin
    const safety = 0.8; // require 20% headroom
    return estNetMs < this.downgradeMs * safety;
  }

  _bump(delta, why) {
    const old = this.profile;
    const next = Math.max(this.minProfile, Math.min(this.maxProfile, this.profile + delta));
    if (next === old) return;
    this.profile = next;
    const now = performance.now();
    if (delta < 0) this.lastUpAt = now; else this.lastDownAt = now;
    if (this.debug) console.log(`[ABR] profile ${old} -> ${this.profile} (${why})`);
  }

  _median(a) {
    const s = a.slice().sort((x, y) => x - y);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
}
