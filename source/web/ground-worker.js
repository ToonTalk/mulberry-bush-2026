// Grows the ground texture with render.py's recipe (make_ground): layered
// bicubic value noise mixing soil and moss, per-pixel grain, 90,000 grass and
// twig strokes, 900 pebbles. Same recipe, its own random numbers.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGround(N, seed) {
  const rnd = mulberry32(seed);
  const cub = (x) => {            // Keys cubic, a = -0.5 (Pillow's bicubic)
    x = Math.abs(x);
    if (x < 1) return (1.5 * x - 2.5) * x * x + 1;
    if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
  };
  // value noise: a random grid of n x n, bicubically enlarged by `cell`
  function noise(cell) {
    const n = Math.floor(N / cell) + 3;
    const g = new Float32Array(n * n);
    for (let i = 0; i < g.length; i++) g[i] = rnd();
    const idx = new Int32Array(N * 4), wt = new Float32Array(N * 4);
    for (let x = 0; x < N; x++) {
      const s = (x + 0.5) / cell - 0.5;
      const i0 = Math.floor(s) - 1;
      let ws = 0;
      for (let k = 0; k < 4; k++) {
        const w = cub(s - (i0 + k));
        idx[x * 4 + k] = Math.min(n - 1, Math.max(0, i0 + k));
        wt[x * 4 + k] = w; ws += w;
      }
      for (let k = 0; k < 4; k++) wt[x * 4 + k] /= ws;
    }
    const rows = new Float32Array(n * N);
    for (let r = 0; r < n; r++) {
      for (let x = 0; x < N; x++) {
        let v = 0;
        for (let k = 0; k < 4; k++) v += g[r * n + idx[x * 4 + k]] * wt[x * 4 + k];
        rows[r * N + x] = v;
      }
    }
    const out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) {
      const a0 = idx[y * 4] * N, a1 = idx[y * 4 + 1] * N, a2 = idx[y * 4 + 2] * N, a3 = idx[y * 4 + 3] * N;
      const w0 = wt[y * 4], w1 = wt[y * 4 + 1], w2 = wt[y * 4 + 2], w3 = wt[y * 4 + 3];
      const o = y * N;
      for (let x = 0; x < N; x++) out[o + x] = rows[a0 + x] * w0 + rows[a1 + x] * w1 + rows[a2 + x] * w2 + rows[a3 + x] * w3;
    }
    return out;
  }
  const NN = N * N;
  const mix = (layers) => {            // weighted sum of noise layers, one temp array at a time
    const out = noise(layers[0][0]);
    for (let i = 0; i < NN; i++) out[i] *= layers[0][1];
    for (let j = 1; j < layers.length; j++) {
      const tmp = noise(layers[j][0]), w = layers[j][1];
      for (let i = 0; i < NN; i++) out[i] += tmp[i] * w;
    }
    return out;
  };
  const n1 = mix([[300, 0.5], [120, 0.3], [45, 0.2]]);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < NN; i++) { const v = n1[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  for (let i = 0; i < NN; i++) n1[i] = (n1[i] - lo) / (hi - lo);
  const n2 = mix([[16, 0.6], [6, 0.4]]);
  const n3 = mix([[3, 0.5], [2, 0.5]]);
  const soil = [0.33, 0.27, 0.24], moss = [0.27, 0.31, 0.21];
  const px = new Float32Array(NN * 3);
  for (let i = 0; i < NN; i++) {
    let m = Math.min(1, Math.max(0, (n1[i] - 0.25) / 0.5));
    m = m * m * (3 - 2 * m);
    const k = (0.95 + 0.08 * n1[i]) * (0.86 + 0.22 * n2[i]) * (0.74 + 0.42 * n3[i]) * (0.86 + 0.28 * rnd());
    for (let c = 0; c < 3; c++) px[i * 3 + c] = Math.min(1, Math.max(0, (soil[c] * (1 - m) + moss[c] * m) * k)) * 255;
  }
  const blend = (x, y, r, g, b, a) => {
    if (x < 0 || y < 0 || x >= N || y >= N) return;
    const o = (y * N + x) * 3;
    px[o] += (r - px[o]) * a; px[o + 1] += (g - px[o + 1]) * a; px[o + 2] += (b - px[o + 2]) * a;
  };
  const U = (a, b) => a + (b - a) * rnd();
  // grass blades and dark twigs: one-pixel lines with alpha
  for (let s = 0; s < 90000; s++) {
    const x = U(0, N), y = U(0, N), a = U(0, 2 * Math.PI), ln = U(2, 7);
    let c;
    if (rnd() < 0.5) c = [U(90, 140), U(110, 160), U(60, 95), U(50, 130)];
    else c = [U(15, 35), U(15, 35), U(10, 25), U(60, 140)];
    // like Pillow's one-pixel line: integer end points, one pixel per step of the longer axis
    const xa = Math.round(x), ya = Math.round(y);
    const xb = Math.round(x + ln * Math.cos(a)), yb = Math.round(y + ln * Math.sin(a));
    const steps = Math.max(Math.abs(xb - xa), Math.abs(yb - ya));
    for (let i = 0; i <= steps; i++) {
      const f = steps ? i / steps : 0;
      blend(Math.round(xa + (xb - xa) * f), Math.round(ya + (yb - ya) * f), c[0], c[1], c[2], c[3] / 255);
    }
  }
  // pebbles
  for (let s = 0; s < 900; s++) {
    const x = U(0, N), y = U(0, N), r = U(0.8, 2.2), v = U(90, 130);
    const ry = r * 0.8;
    for (let yy = Math.floor(y - ry); yy <= Math.ceil(y + ry); yy++) {
      for (let xx = Math.floor(x - r); xx <= Math.ceil(x + r); xx++) {
        const dx = (xx + 0.5 - x) / r, dy = (yy + 0.5 - y) / ry;
        if (dx * dx + dy * dy <= 1) blend(xx, yy, v, v - 8, v - 16, 140 / 255);
      }
    }
  }
  const rgba = new Uint8ClampedArray(NN * 4);
  for (let i = 0; i < NN; i++) {
    rgba[i * 4] = px[i * 3]; rgba[i * 4 + 1] = px[i * 3 + 1]; rgba[i * 4 + 2] = px[i * 3 + 2]; rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

if (typeof self !== "undefined" && typeof module === "undefined") {
  self.onmessage = (e) => {
    const { N, seed } = e.data;
    const px = makeGround(N, seed);
    self.postMessage({ N, px }, [px.buffer]);
  };
}
if (typeof module !== "undefined") module.exports = { makeGround };
