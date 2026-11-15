// Synthesises the soundtrack: a port of audio.py's instruments, drone and
// textures. The score itself (every note, footstep, tink, plop, cricket and
// bird, with its time, gain and pan) comes from web_export.py, which recorded
// it from audio.py, so the browser plays the same events. Only the noise
// samples and the random phases of partials are this file's own.
//
// The timeline is split into windows so several workers can share the work:
// a window synthesises the events that start inside it (their tails spill past
// its end) and the drone and textures inside it. Needs TL (timeline.js).
const AW = (() => {
  const SR = 48000;
  const PI2 = 2 * Math.PI;
  const TAIL = 6.0;                      // longest event: 3.5 s note + 1.6 s ring
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  let rnd = mulberry32(76);
  // gaussian noise: a table of 2^20 Box-Muller samples, read from random places
  const GN = 1 << 20, GM = GN - 1;
  const GT = new Float32Array(GN);
  let gi = 0, gleft = 0;
  function fillGauss() {
    for (let i = 0; i < GN; i += 2) {
      let u = 0;
      while (u === 0) u = rnd();
      const v = rnd(), r = Math.sqrt(-2 * Math.log(u));
      GT[i] = r * Math.cos(PI2 * v); GT[i + 1] = r * Math.sin(PI2 * v);
    }
  }
  function gauss() {
    if (gleft-- <= 0) { gi = Math.floor(rnd() * GN); gleft = 4096; }
    return GT[(gi++) & GM];
  }
  const SCR = new Float64Array(SR * 8);
  const ENV = new Float64Array(SR * 8);

  // SCR[0..n) = sum of A r^i sin(w i + phase): one damped resonator per partial,
  // all advanced together, sample by sample
  const PY1 = new Float64Array(32), PY2 = new Float64Array(32), PC = new Float64Array(32), PR = new Float64Array(32);
  function partials(n, list) {
    const P = list.length;
    for (let p = 0; p < P; p++) {
      const [amp, freq, phase, dec] = list[p];
      const w = PI2 * freq / SR, r = Math.exp(-dec / SR);
      PY1[p] = amp / r * Math.sin(phase - w);            // y[-1]
      PY2[p] = amp / (r * r) * Math.sin(phase - 2 * w);  // y[-2]
      PC[p] = 2 * r * Math.cos(w); PR[p] = r * r;
    }
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let p = 0; p < P; p++) {
        const y = PC[p] * PY1[p] - PR[p] * PY2[p];
        PY2[p] = PY1[p]; PY1[p] = y;
        acc += y;
      }
      SCR[i] = acc;
    }
  }
  // env_adsr of audio.py, written into ENV[0..n)
  function envADSR(n, a, r, sus) {
    const na = Math.max(1, Math.floor(a * SR)), ns = Math.floor(sus * SR), nr = Math.max(1, Math.floor(r * SR));
    for (let i = 0; i < n; i++) ENV[i] = 1;
    for (let i = 0; i < Math.min(na, n); i++) ENV[i] = na > 1 ? i / (na - 1) : 0;
    if (ns < n) {
      const k = Math.exp(-3.0 / nr);
      let g = 1;
      for (let i = ns; i < n; i++) { ENV[i] *= g; g *= k; }
    }
  }
  // a sine carrier with vibrato: (c, s) is rotated by each sample's phase step;
  // the step's cos/sin come from short Taylor series (the step is < 0.2 rad)
  function vibratoCarrier(n, f, depth, rate, onset, ramp, write) {
    let c = 1, s = 0;                      // phase 0 before the first sample
    const wv = PI2 * rate / SR, cvv = Math.cos(wv), svv = Math.sin(wv);
    let vc = 1, vs = 0;                    // vibrato oscillator, sin(wv * i)
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      const ramp01 = Math.min(1, Math.max(0, (t - onset) / ramp));
      const d = PI2 * f * (1 + depth * vs * ramp01) / SR;
      const d2 = d * d;
      const cd = 1 - d2 / 2 + d2 * d2 / 24, sd = d - d * d2 / 6 + d * d2 * d2 / 120;
      const nc = c * cd - s * sd, ns = s * cd + c * sd;
      c = nc; s = ns;
      if ((i & 511) === 511) { const k = 1 / Math.sqrt(c * c + s * s); c *= k; s *= k; }
      write(i, s, c);
      const nvc = vc * cvv - vs * svv; vs = vs * cvv + vc * svv; vc = nvc;
    }
  }

  // ------------------------------------------------------------ instruments
  function pluck(f, dur) {
    const n = Math.floor((dur + 1.6) * SR);
    const list = [];
    for (let p = 1; p < 16; p++) {
      const fn = p * f * Math.sqrt(1 + 0.0003 * p * p);
      if (fn > 14000) break;
      list.push([Math.abs(Math.sin(Math.PI * p * 0.17)) / Math.pow(p, 1.05), fn, rnd() * 6.28, 1.4 + 0.75 * p * Math.sqrt(f / 440)]);
    }
    partials(n, list);
    const na = 0.003 * SR, itd = Math.floor((dur + 0.35) * SR) + 1;
    for (let i = 0; i < Math.min(n, itd); i++) SCR[i] *= Math.min(1, i / na) * 0.9;
    let g = Math.exp(-(itd / SR - dur - 0.35) * 5) * 0.9;
    const k = Math.exp(-5 / SR);
    for (let i = itd; i < n; i++) { SCR[i] *= g; g *= k; }
    return n;
  }
  function flute(f, dur) {
    const n = Math.floor((dur + 0.3) * SR);
    envADSR(n, 0.07, 0.12, dur);
    let br = 0, be = 0.6;
    const kb = Math.exp(-1 / SR / 0.12);
    vibratoCarrier(n, f, 0.0045, 5.1, 0.15, 0.3, (i, s1, c1) => {
      const s = s1 + 0.20 * (2 * s1 * c1) + 0.06 * (3 * s1 - 4 * s1 * s1 * s1);
      br = 0.08 * gauss() + 0.92 * br;
      SCR[i] = (s + br * (be + 0.25) * 0.8) * ENV[i] * 0.8;
      be *= kb;
    });
    return n;
  }
  function reed(f, dur) {
    const n = Math.floor((dur + 0.25) * SR);
    envADSR(n, 0.09, 0.1, dur);
    const A = new Float64Array(21);
    let P = 0;
    for (let p = 1; p < 22; p++) {
      if (p * f > 9000) break;
      A[P++] = ((p % 2) ? 1.0 : 0.32) / p / (1 + (p * f / 2200) ** 2);
    }
    vibratoCarrier(n, f, 0.003, 4.4, 0.2, 0.4, (i, s1, c1) => {
      const c2 = 2 * c1;
      let sp = 0, sc = s1, acc = A[0] * s1;
      for (let p = 1; p < P; p++) { const sn = c2 * sc - sp; acc += A[p] * sn; sp = sc; sc = sn; }
      SCR[i] = acc * ENV[i] * 0.9;
    });
    return n;
  }
  function marimba(f) {
    const n = Math.floor(2.2 * SR);
    partials(n, [[1.0, f, 0, 2.0 + f / 700], [0.32, f * 3.98, 0, 9], [0.10, f * 9.9, 0, 18], [0.06, f * 2.0, 0, 6]]);
    let y = 0, g = 0.25;
    const nc = Math.floor(0.08 * SR), kc = Math.exp(-1 / SR / 0.004);
    for (let i = 0; i < nc; i++) { y = 0.3 * gauss() + 0.6 * y; SCR[i] += y * g; g *= kc; }
    const na = 0.002 * SR;
    for (let i = 0; i < na; i++) SCR[i] *= i / na;
    return n;
  }
  function bell(f, L) {
    const n = Math.floor(L * SR);
    partials(n, [[1.0, f, 0, 3.5], [0.4, f * 2.76, 0, 7], [0.2, f * 5.4, 0, 12]]);
    const na = 0.001 * SR;
    for (let i = 0; i < na; i++) SCR[i] *= i / na;
    return n;
  }
  function plop(f) {
    const n = Math.floor(0.35 * SR);
    let ph = 0, e1 = 1, e2 = 1;
    const k1 = Math.exp(-1 / SR / 0.018), k2 = Math.exp(-1 / SR / 0.07), na = 0.0015 * SR;
    for (let i = 0; i < n; i++) {
      ph += PI2 * f * (1 + 1.4 * e1) / SR;
      SCR[i] = Math.sin(ph) * e2 * Math.min(1, i / na);
      e1 *= k1; e2 *= k2;
    }
    return n;
  }
  function thump(bright) {
    const n = Math.floor(0.16 * SR);
    let ph = 0, lp = 0, e1 = 1, e2 = 1, e3 = 0.25 + 0.3 * bright;
    const k1 = Math.exp(-1 / SR / 0.01), k2 = Math.exp(-1 / SR / 0.035), k3 = Math.exp(-1 / SR / (0.012 + 0.01 * bright));
    for (let i = 0; i < n; i++) {
      ph += PI2 * 70 * (1 + 0.8 * e1) / SR;
      const x = gauss();
      lp = 0.1 * x + 0.9 * lp;
      SCR[i] = Math.sin(ph) * e2 + (x - lp) * e3;
      e1 *= k1; e2 *= k2; e3 *= k3;
    }
    return n;
  }
  function cricket() {
    const L = Math.floor(0.022 * SR), G = Math.floor(0.018 * SR);
    let i = 0;
    for (let c = 0; c < 3; c++) {
      for (let j = 0; j < L; j++) {
        const t = j / SR, e = Math.sin(Math.PI * t / 0.022);
        SCR[i++] = Math.sin(PI2 * 4400 * t) * e * e;
      }
      for (let j = 0; j < G; j++) SCR[i++] = 0;
    }
    return i;
  }
  function chirp(syll, i0, cnt) {
    let i = 0;
    for (let s = i0; s < i0 + cnt; s++) {
      const L = syll[s * 5], f0 = syll[s * 5 + 1], f1 = syll[s * 5 + 2], vib = syll[s * 5 + 3], gap = syll[s * 5 + 4];
      const n = Math.floor(L * SR);
      let ph = 0;
      for (let j = 0; j < n; j++) {
        const t = j / SR;
        ph += PI2 * (f0 + (f1 - f0) * (t / L) + 300 * Math.sin(PI2 * vib * t)) / SR;
        const e = Math.sin(Math.PI * t / L);
        SCR[i++] = Math.sin(ph) * e * e;
      }
      const g = Math.floor(gap * SR);
      for (let j = 0; j < g; j++) SCR[i++] = 0;
    }
    return i;
  }
  function boom() {
    const n = Math.floor(2.5 * SR);
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      SCR[i] = Math.sin(PI2 * 46 * t * (1 + 0.5 * Math.exp(-t / 0.05))) * Math.exp(-t / 0.6);
    }
    return n;
  }

  // ------------------------------------------------------------ helpers
  const smooth = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));
  const mtof = (m) => 440.0 * Math.pow(2, (m - 69) / 12);
  const panGains = (p) => { const a = (Math.min(1, Math.max(-1, p)) + 1) * Math.PI / 4; return [Math.cos(a), Math.sin(a)]; };

  // bar number of samples s0..s1, from coarse clock lookups
  function barsOfSamples(s0, s1) {
    const n = s1 - s0, out = new Float32Array(n), K = 32;
    let b0 = TL.B_of_t(s0 / SR);
    for (let i = 0; i < n; i += K) {
      const b1 = TL.B_of_t((s0 + i + K) / SR);
      const m = Math.min(K, n - i);
      for (let j = 0; j < m; j++) out[i + j] = b0 + (b1 - b0) * (j / K);
      b0 = b1;
    }
    return out;
  }
  // band-limited gaussian noise: 4th-order high-pass and low-pass, scaled to the
  // level of audio.py's brick-wall FFT band (std = sqrt(band / nyquist))
  function bandNoise(lo, hi, n) {
    const out = new Float32Array(n);
    const biq = (type, fc) => {
      const w = PI2 * fc / SR, cw = Math.cos(w), al = Math.sin(w) / (2 * Math.SQRT1_2);
      let b0, b1, b2;
      if (type === "lp") { b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2; } else { b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2; }
      const a0 = 1 + al;
      return [b0 / a0, b1 / a0, b2 / a0, (-2 * cw) / a0, (1 - al) / a0];
    };
    const stages = [biq("hp", lo), biq("hp", lo), biq("lp", hi), biq("lp", hi)];
    const st = stages.map(() => [0, 0, 0, 0]);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      let x = gauss();
      for (let s = 0; s < 4; s++) {
        const c = stages[s], z = st[s];
        const y = c[0] * x + c[1] * z[0] + c[2] * z[1] - c[3] * z[2] - c[4] * z[3];
        z[1] = z[0]; z[0] = x; z[3] = z[2]; z[2] = y;
        x = y;
      }
      out[i] = x;
      acc += x * x;
    }
    const k = Math.sqrt((hi - lo) / (SR / 2)) / Math.sqrt(acc / Math.max(1, n));
    for (let i = 0; i < n; i++) out[i] *= k;
    return out;
  }
  function histSmoothed(vals, nb, total, klen) {
    const h = new Float64Array(nb);
    const bw = total / nb;
    for (const v of vals) {
      if (!(v >= 0 && v <= total)) continue;
      h[Math.min(nb - 1, Math.floor(v / bw))]++;
    }
    const lo = Math.floor(klen / 2), hi = klen - 1 - lo;       // numpy convolve(..., "same")
    const out = new Float64Array(nb);
    for (let i = 0; i < nb; i++) {
      let s = 0;
      for (let j = i - lo; j <= i + hi; j++) if (j >= 0 && j < nb) s += h[j];
      out[i] = s / klen;
    }
    return { v: out, x0: 0.05, dx: bw };
  }
  function interpUniform(x, x0, dx, ys) {        // np.interp on a uniform grid
    const f = (x - x0) / dx;
    if (f <= 0) return ys[0];
    const i = Math.floor(f);
    if (i >= ys.length - 1) return ys[ys.length - 1];
    return ys[i] + (ys[i + 1] - ys[i]) * (f - i);
  }

  // ------------------------------------------------------------ one window of the mix
  // D: score data; win: [a, b) seconds; returns { s0, L, R } covering [a, b + TAIL)
  function synthesize(D, win, wi, progress) {
    rnd = mulberry32(76 + 7919 * wi);
    fillGauss();
    const N = Math.floor((TL.TOTAL + 1.0) * SR);
    const s0 = Math.min(N, Math.floor(win[0] * SR)), sEnd = Math.min(N, Math.floor(win[1] * SR));
    const s1 = Math.min(N, Math.floor((win[1] + TAIL) * SR));
    const L = new Float32Array(s1 - s0), R = new Float32Array(s1 - s0);
    const step = (frac, what) => progress && progress(frac, what);

    // ---- the recorded events that start in this window
    const ev = D.events, ne = ev.length / 8;
    for (let e = 0; e < ne; e++) {
      const o = e * 8;
      const t = ev[o + 1];
      const st = Math.trunc(t * SR);
      if (st < s0 || st >= sEnd) continue;
      const kind = Math.round(ev[o]), gain = ev[o + 2], pan = ev[o + 3];
      const p1 = ev[o + 4], p2 = ev[o + 5], cut = ev[o + 6], scale = ev[o + 7];
      let n;
      switch (kind) {
        case 0: n = pluck(p1, p2); break;
        case 1: n = flute(p1, p2); break;
        case 2: n = reed(p1, p2); break;
        case 3: n = marimba(p1, p2); break;
        case 4: n = bell(p1, p2); break;
        case 5: n = plop(p1); break;
        case 6: n = thump(p1); break;
        case 7: n = cricket(); break;
        case 8: n = chirp(D.syll, Math.round(p1), Math.round(p2)); break;
        default: n = boom();
      }
      if (cut >= 0) {
        const nk = Math.floor(Math.max(0, cut - t) * SR);
        for (let i = nk; i < n; i++) SCR[i] *= Math.exp(-(i - nk) / (0.025 * SR));
      }
      const m = Math.min(n, s1 - st);
      const [gl, gr] = panGains(pan);
      const a = gain * scale * gl, b = gain * scale * gr, d = st - s0;
      for (let i = 0; i < m; i++) { const v = SCR[i]; L[d + i] += v * a; R[d + i] += v * b; }
      if (e % 100 === 0) step(0.6 * e / ne, "score");
    }

    // ---- drone: G, E, C, (silence), E, G
    step(0.6, "drone");
    const bAll = barsOfSamples(s0, sEnd);
    const nW = sEnd - s0;
    const segs = [[0, 46, [43, 50, 55]], [46, 64, [40, 47, 52]], [64, 80, [36, 43, 48, 55]],
      [81, 91, [40, 47, 52]], [91, 116, [43, 50, 55, 59]]];
    const ampKeys = [[0, 0], [0.6, 0], [3, 0.5], [10, 0.65], [46, 0.65], [64, 0.75], [79.9, 1.0, "l"],
      [80.04, 0], [81, 0], [83, 0.42], [91, 0.42], [93, 0.6], [109, 0.6], [114.5, 0]];
    const xf = 0.75, EK = 64, TN = 8192;
    const tab = new Float64Array(TN + 1);
    for (const [b0, b1, chord] of segs) {
      const wf = (bb) => smooth((bb - (b0 - xf)) / (2 * xf)) * (1 - smooth((bb - (b1 - xf)) / (2 * xf)));
      let i0 = -1, i1 = -1;
      for (let i = 0; i < nW; i++) if (wf(bAll[i]) > 1e-4) { if (i0 < 0) i0 = i; i1 = i + 1; }
      if (i0 < 0) continue;
      const n = i1 - i0;
      const env = new Float32Array(n);
      for (let i = 0; i < n; i += EK) {
        const e0 = wf(bAll[i0 + i]) * TL.kf(ampKeys, bAll[i0 + i]);
        const j1 = Math.min(n - 1, i + EK), e1 = wf(bAll[i0 + j1]) * TL.kf(ampKeys, bAll[i0 + j1]);
        for (let j = i; j < Math.min(n, i + EK); j++) env[j] = e0 + (e1 - e0) * ((j - i) / EK);
      }
      for (const m of chord) {
        const f = mtof(m);
        const lvl = (0.05 / Math.sqrt(chord.length)) * (m > 40 ? 1.0 : 1.3);
        const lw = PI2 / (7.0 + (m % 5));
        for (const [det, pan] of [[-0.004, -0.5], [0.004, 0.5]]) {
          const ff = f * (1 + det);
          tab.fill(0);                    // one period of sum_p sin(p*phi + 1.3p) / p^1.7
          for (let p = 1; p < 13; p++) {
            if (p * ff > 5000) break;
            const a = 1 / Math.pow(p, 1.7);
            for (let j = 0; j <= TN; j++) tab[j] += a * Math.sin(p * PI2 * j / TN + 1.3 * p);
          }
          const [gl, gr] = panGains(pan * 0.6);
          const cyc = ff / SR;
          const sA = s0 + i0;             // absolute sample index of the first sample
          let ph = sA * cyc; ph -= Math.floor(ph);
          for (let i = 0; i < n; i += EK) {
            const m1 = Math.min(n, i + EK);
            const l0 = (0.75 + 0.25 * Math.sin(lw * (sA + i) / SR + m)) * lvl;
            const l1 = (0.75 + 0.25 * Math.sin(lw * (sA + i + EK) / SR + m)) * lvl;
            for (let j = i; j < m1; j++) {
              const x = ph * TN, k = x | 0;
              const w = tab[k] + (tab[k + 1] - tab[k]) * (x - k);
              const v = w * (l0 + (l1 - l0) * ((j - i) / EK)) * env[j];
              L[i0 + j] += v * gl; R[i0 + j] += v * gr;
              ph += cyc; if (ph >= 1) ph -= 1;
            }
          }
        }
      }
    }

    // ---- textures: rustle of unfurling leaves, the whirlwind, flutter of torn leaves
    step(0.75, "wind");
    const nb = Math.floor(TL.TOTAL * 10);
    const dens = histSmoothed(D.leafBorn, nb, TL.TOTAL, 8);
    const ldens = histSmoothed(D.leafDet, nb, TL.TOTAL, 10);
    const gR = Array.from(D.gateRust, (v) => v / 255), gF = Array.from(D.gateFlutter, (v) => v / 255);
    const range = (fn) => {             // the part of this window where fn > 0
      let a = -1, z = -1;
      for (let i = 0; i < nW; i += 240) if (fn(i) > 0) { if (a < 0) a = i; z = i; }
      return a < 0 ? null : [Math.max(0, a - 240), Math.min(nW, z + 480)];
    };
    const densAt = (i) => interpUniform((s0 + i) / SR, dens.x0, dens.dx, dens.v);
    const ldensAt = (i) => interpUniform((s0 + i) / SR, ldens.x0, ldens.dx, ldens.v);
    let rg = range(densAt);
    if (rg) {
      const nz = bandNoise(2500, 8000, rg[1] - rg[0]);
      for (let i = rg[0]; i < rg[1]; i++) {
        const t = (s0 + i) / SR;
        const g = Math.pow(interpUniform(t, 0, 0.12, gR), 3);
        const v = nz[i - rg[0]] * Math.min(1, Math.max(0, densAt(i) / 12)) * 0.035 * (0.2 + 1.6 * g);
        L[i] += v * (0.8 + 0.2 * Math.sin(t * 1.3)); R[i] += v * (0.8 + 0.2 * Math.cos(t * 1.1));
      }
    }
    const windKeys = [[0, 0], [64, 0], [68, 0.25], [79.9, 1.0, "i"], [80.05, 0.0], [200, 0]];
    rg = range((i) => TL.kf(windKeys, bAll[i]));
    if (rg) {
      const nz = bandNoise(150, 1800, rg[1] - rg[0]);
      // the sweep's phase integrates from the start of the film
      let cs = 0;
      const sStart = s0 + rg[0];
      const tw0 = TL.t_of_B(64);
      if (sStart / SR > tw0) {
        const sw = Math.floor(tw0 * SR);
        cs = 0.2 * sw;
        for (let i = sw; i < sStart; i += EK) cs += (0.2 + 0.6 * TL.kf(windKeys, TL.B_of_t(i / SR))) * Math.min(EK, sStart - i);
      } else cs = 0.2 * sStart;
      for (let i = rg[0]; i < rg[1]; i++) {
        const t = (s0 + i) / SR, wa = TL.kf(windKeys, bAll[i]);
        cs += 0.2 + 0.6 * wa;
        const v = nz[i - rg[0]] * wa * (0.6 + 0.4 * Math.sin(PI2 * cs / SR)) * 0.12;
        L[i] += v * (0.7 + 0.3 * Math.sin(t * 2.1)); R[i] += v * (0.7 + 0.3 * Math.cos(t * 2.1));
      }
    }
    step(0.9, "wind");
    rg = range(ldensAt);
    if (rg) {
      const nz = bandNoise(1500, 7000, rg[1] - rg[0]);
      for (let i = rg[0]; i < rg[1]; i++) {
        const t = (s0 + i) / SR;
        const g = Math.pow(interpUniform(t, 0, 0.07, gF), 4);
        const v = nz[i - rg[0]] * Math.min(1, Math.max(0, ldensAt(i) / 15)) * 0.04 * (0.15 + 2.0 * g);
        L[i] += v;
        if (i + 2400 < L.length) R[i + 2400] += v;
      }
    }
    step(1, "done");
    return { s0, L, R };
  }

  // ------------------------------------------------------------ the room
  // radix-2 complex FFT, in place; inverse is scaled by 1/n
  function makeFFT(n) {
    const bits = Math.round(Math.log2(n));
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      rev[i] = r;
    }
    const cs = new Float64Array(n / 2), sn = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cs[i] = Math.cos(PI2 * i / n); sn[i] = Math.sin(PI2 * i / n); }
    return function (re, im, inverse) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
      }
      const sg = inverse ? -1 : 1;
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const l = j + half, c = cs[k], s_ = sg * sn[k];
            const tr = re[l] * c + im[l] * s_, ti = im[l] * c - re[l] * s_;
            re[l] = re[j] - tr; im[l] = im[j] - ti;
            re[j] += tr; im[j] += ti;
          }
        }
      }
      if (inverse) { const k = 1 / n; for (let i = 0; i < n; i++) { re[i] *= k; im[i] *= k; } }
    };
  }
  // dry + 0.42 * (dry convolved with the room), both channels at once: the pair is
  // packed as L + iR, split by conjugate symmetry, filtered, and packed back
  function addRoom(L, R, irL, irR) {
    const n = L.length, m = irL.length, F = 1 << 18, hop = F - m + 1;
    const fft = makeFFT(F);
    const re = new Float64Array(F), im = new Float64Array(F);
    for (let i = 0; i < m; i++) { re[i] = irL[i]; im[i] = irR[i]; }
    fft(re, im, false);
    const HLr = new Float64Array(F), HLi = new Float64Array(F), HRr = new Float64Array(F), HRi = new Float64Array(F);
    for (let k = 0; k < F; k++) {
      const k2 = (F - k) & (F - 1);
      const xr = re[k], xi = im[k], zr = re[k2], zi = im[k2];
      HLr[k] = (xr + zr) / 2; HLi[k] = (xi - zi) / 2; HRr[k] = (xi + zi) / 2; HRi[k] = -(xr - zr) / 2;
    }
    const oL = new Float32Array(n + m - 1), oR = new Float32Array(n + m - 1);
    oL.set(L); oR.set(R);
    const yRe = new Float64Array(F), yIm = new Float64Array(F);
    for (let s0 = 0; s0 < n; s0 += hop) {
      const len = Math.min(hop, n - s0);
      re.fill(0); im.fill(0);
      for (let i = 0; i < len; i++) { re[i] = L[s0 + i]; im[i] = R[s0 + i]; }
      fft(re, im, false);
      for (let k = 0; k < F; k++) {
        const k2 = (F - k) & (F - 1);
        const xr = re[k], xi = im[k], zr = re[k2], zi = im[k2];
        const lr = (xr + zr) / 2, li = (xi - zi) / 2, rr = (xi + zi) / 2, ri = -(xr - zr) / 2;
        const ylr = lr * HLr[k] - li * HLi[k], yli = lr * HLi[k] + li * HLr[k];
        const yrr = rr * HRr[k] - ri * HRi[k], yri = rr * HRi[k] + ri * HRr[k];
        yRe[k] = ylr - yri; yIm[k] = yli + yrr;
      }
      fft(yRe, yIm, true);
      const lim = Math.min(F, n + m - 1 - s0);
      for (let i = 0; i < lim; i++) { oL[s0 + i] += 0.42 * yRe[i]; oR[s0 + i] += 0.42 * yIm[i]; }
    }
    return [oL, oR];
  }

  // the room's impulse response: 3.2 s of decaying, low-passed noise per channel, unit energy
  function impulse() {
    rnd = mulberry32(3); fillGauss();
    const Lr = Math.floor(3.2 * SR);
    return [0.012, 0.017].map((pre) => {
      const a = new Float32Array(Lr);
      let y = 0, e = 0;
      for (let i = 0; i < Lr; i++) { y = 0.35 * gauss() * Math.exp(-i / SR / 0.75) + 0.65 * y; a[i] = y; }
      a.fill(0, 0, Math.floor(pre * SR));
      for (let i = 0; i < Lr; i++) e += a[i] * a[i];
      const k = 1 / Math.sqrt(e);
      for (let i = 0; i < Lr; i++) a[i] *= k;
      return a;
    });
  }
  return { synthesize, impulse, addRoom, SR, TAIL };
})();

if (typeof self !== "undefined" && typeof module === "undefined") {
  self.onmessage = (e) => {
    const { data, win, wi } = e.data;
    let last = 0;
    const tA = Date.now();
    const out = AW.synthesize(data, win, wi, (f, what) => {
      const now = Date.now();
      if (now - last > 150) { last = now; self.postMessage({ progress: f * 0.8, what }); }
    });
    const tB = Date.now();
    const [irL, irR] = AW.impulse();
    const [L, R] = AW.addRoom(out.L, out.R, irL, irR);
    self.postMessage({ done: true, s0: out.s0, L, R, ms: [tB - tA, Date.now() - tB] }, [L.buffer, R.buffer]);
  };
}
if (typeof module !== "undefined") module.exports = AW;
