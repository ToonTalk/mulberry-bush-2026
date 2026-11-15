// Shared clock for picture and sound: a direct port of timeline.py.
// Everything is driven by one musical clock measured in bars of 6/8.
// World coordinates: origin = the bush, y up, units ~ screen pixels at zoom 1.
const TL = (() => {
  const PI = Math.PI;
  const FPS = 30;
  const SECTIONS = [
    ["dark", 2, 1.5, 1.5], ["enter", 8, 1.5, 1.5], ["grow", 20, 1.5, 1.5], ["leaf", 16, 1.5, 1.5],
    ["fruit", 18, 1.5, 1.5], ["fall", 16, 1.5, 0.8], ["hold", 1, 2.2, 2.2], ["night", 10, 1.8, 1.8],
    ["dawn", 14, 1.8, 1.8], ["title", 10, 2.0, 2.0],
  ];
  const SB = {};
  let nb = 0;
  for (const [name, n] of SECTIONS) { SB[name] = nb; nb += n; }
  const NBARS = nb;
  const SUB = 480;
  const NG = NBARS * SUB + 1;
  const step = NBARS / (NG - 1);
  const B = new Float64Array(NG);
  for (let i = 0; i < NG; i++) B[i] = i * step;
  B[NG - 1] = NBARS;
  const dur = new Float64Array(NG);
  for (const [name, n, d0, d1] of SECTIONS) {
    const b0 = SB[name];
    for (let i = 0; i < NG; i++) {
      if (B[i] >= b0 && B[i] <= b0 + n) dur[i] = d0 + (d1 - d0) * ((B[i] - b0) / n);
    }
  }
  const T = new Float64Array(NG);
  for (let i = 1; i < NG; i++) T[i] = T[i - 1] + (dur[i - 1] + dur[i]) * 0.5 * (B[i] - B[i - 1]);
  const TOTAL = T[NG - 1];

  // interpolation on the uniform bar grid
  function onGrid(arr, b) {
    if (b <= 0) return arr[0];
    const x = b / step;
    const i = Math.floor(x);
    if (i >= NG - 1) return arr[NG - 1];
    const f = x - i;
    return arr[i] + (arr[i + 1] - arr[i]) * f;
  }
  const t_of_B = (b) => onGrid(T, b);
  function B_of_t(t) {
    if (t <= 0) return 0;
    if (t >= TOTAL) return NBARS;
    let lo = 0, hi = NG - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
    return B[lo] + (B[hi] - B[lo]) * (t - T[lo]) / (T[hi] - T[lo]);
  }
  const ST = {};
  for (const k in SB) ST[k] = t_of_B(SB[k]);

  const clip01 = (u) => (u < 0 ? 0 : u > 1 ? 1 : u);
  function smooth(u) { u = clip01(u); return u * u * (3 - 2 * u); }

  // keyframes: [bar, value, ease?] with ease s (smooth, default), i (in), o (out), l (linear)
  function kf(keys, b) {
    const n = keys.length;
    let c = 0;
    while (c < n && keys[c][0] <= b) c++;
    let i = c - 1;
    if (i < 0) i = 0;
    if (i > n - 2) i = n - 2;
    const x0 = keys[i][0], x1 = keys[i + 1][0];
    let u = clip01((b - x0) / Math.max(x1 - x0, 1e-9));
    const e = keys[i][2] || "s";
    if (e === "s") u = u * u * (3 - 2 * u);
    else if (e === "i") u = u * u;
    else if (e === "o") u = 1 - (1 - u) * (1 - u);
    return keys[i][1] + (keys[i + 1][1] - keys[i][1]) * u;
  }

  // ------------------------------------------------------------ dancers
  const R0 = 345.0;
  const PHI = [0, 1, 2, 3].map((k) => PI / 4 + k * PI / 2);
  const ENTER = [2.0, 4.0, 6.0, 8.0];
  const EXIT = [97.0, 103.0, 101.0, 99.0];
  const CORNER_R = 880.0;
  const RHO_EVEN = [[0, 0.0], [10.4, 0.0], [12.2, 1 / 8], [46, 1 / 8], [48, 1 / 16], [64, 1 / 16, "i"], [80, 0.30, "l"],
    [80.22, 0.0], [81.6, 0.0], [83.2, 1 / 16], [91, 1 / 16], [95, 0.0]];
  const RHO_ODD = [[0, 0.0], [10.4, 0.0], [12.2, 1 / 8], [30, 1 / 8], [32, -1 / 8], [46, -1 / 8], [48, 1 / 16], [64, 1 / 16, "i"],
    [80, 0.30, "l"], [80.22, 0.0], [81.6, 0.0], [83.2, 1 / 16], [91, 1 / 16], [95, 0.0]];

  const ANG = [], RAD = [];
  for (let k = 0; k < 4; k++) {
    const keys = k % 2 === 0 ? RHO_EVEN : RHO_ODD;
    const rho = new Float64Array(NG);
    for (let i = 0; i < NG; i++) rho[i] = kf(keys, B[i]);
    const a = new Float64Array(NG);
    let acc = 0;
    a[0] = PHI[k];
    for (let i = 1; i < NG; i++) {
      acc += (rho[i - 1] + rho[i]) * 0.5 * (B[i] - B[i - 1]);
      a[i] = PHI[k] + 2 * PI * acc;
    }
    ANG.push(a);
  }
  // align the stopping positions at dawn with the four diagonals
  let i95 = 0;
  while (i95 < NG && B[i95] < 95.0) i95++;
  const a95 = ANG[0][i95] - PHI[0];
  const md = (x, m) => ((x % m) + m) % m;
  const delta = -(md(a95 + PI / 4, PI / 2) - PI / 4);
  for (let i = 0; i < NG; i++) {
    const c = delta * smooth((B[i] - 88.0) / 7.0);
    for (let k = 0; k < 4; k++) ANG[k][i] += c;
  }
  const baseKeys = [[0, R0], [64, R0], [66, R0 + 15], [80, R0 + 35, "l"], [80.22, R0 + 35], [81.5, R0 + 10],
    [83.5, R0], [200, R0]];
  for (let k = 0; k < 4; k++) {
    const r = new Float64Array(NG);
    for (let i = 0; i < NG; i++) r[i] = kf(baseKeys, B[i]);
    RAD.push(r);
  }
  const wvKeys = [[0, 0], [30.5, 0], [32.5, 1], [45, 1], [46.5, 0]];
  const ioAllKeys = [[0, 0], [46.5, 0], [48, 1], [55, 1], [56, 0], [200, 0]];
  const ioAltKeys = [[0, 0], [55, 0], [56, 1], [63, 1], [64.5, 0], [200, 0]];
  const pyRound = (x) => {            // numpy rounds halves to even
    const r = Math.round(x);
    return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
  };
  for (let i = 0; i < NG; i++) {
    const wv = kf(wvKeys, B[i]);
    if (wv > 0) {
      for (const a of [0, 2]) {
        for (const bb of [1, 3]) {
          const s = (a + bb) % 4 === 1 ? 1.0 : -1.0;
          const D = ANG[a][i] - ANG[bb][i];
          const x = D - 2 * PI * pyRound(D / (2 * PI));
          const bump = Math.exp(-((x / 0.6) ** 2)) * wv;
          RAD[a][i] += 50.0 * s * bump;
          RAD[bb][i] -= 50.0 * s * bump;
        }
      }
    }
    const ia = kf(ioAllKeys, B[i]), il = kf(ioAltKeys, B[i]);
    for (let k = 0; k < 4; k++) {
      RAD[k][i] -= 78 * ia * (0.5 - 0.5 * Math.cos(2 * PI * (B[i] - 48) / 4));
      RAD[k][i] -= 78 * il * (0.5 - 0.5 * Math.cos(2 * PI * (B[i] - 56) / 4 + PI * (k % 2)));
    }
  }

  const angle = (k, b) => onGrid(ANG[k], b);
  const radius = (k, b) => onGrid(RAD[k], b);
  const COUT = [0, 1, 2, 3].map((k) => {
    const th = angle(k, EXIT[k]);
    return [CORNER_R * Math.cos(th), CORNER_R * Math.sin(th)];
  });

  // world position of dancer k at bar b, written into out[0], out[1]
  function pos(k, b, out) {
    const th = angle(k, b), r = radius(k, b);
    let px = r * Math.cos(th), py = r * Math.sin(th);
    const cx = CORNER_R * Math.cos(PHI[k]), cy = CORNER_R * Math.sin(PHI[k]);
    const e = smooth((b - ENTER[k]) / 3.2);
    px = cx + (px - cx) * e; py = cy + (py - cy) * e;
    const x = clip01((b - EXIT[k]) / 3.4);
    const e2 = x * x * (3 - 2 * x) * 0.5 + x * x * 0.5;
    out[0] = px + (COUT[k][0] - px) * e2;
    out[1] = py + (COUT[k][1] - py) * e2;
    return out;
  }

  const intKeys = [[0, 1.0], [79, 1.12], [80.15, 0.30], [81.2, 0.30], [83, 0.62], [91, 0.62], [99, 0.48], [200, 0.48]];
  const intensity = (k, b) => kf(intKeys, b) * (1 - smooth((b - EXIT[k] - 0.5) / 3.0));
  const hlKeys = [[0, 150], [79.8, 155], [80.4, 62], [81.6, 62], [83.5, 135], [91, 135], [97, 150], [200, 150]];
  const light_height = (b) => kf(hlKeys, b);
  const rjKeys = [[0, 0], [10.5, 0], [12.5, 1], [30, 1], [30.7, 0], [46.8, 0], [48.6, 0.75], [64, 0.75],
    [66, 1.0], [80, 1.0], [80.12, 0.0], [200, 0]];
  const ring_join = (b) => kf(rjKeys, b);
  const tmKeys = [[0, 0.7], [30, 0.7], [31.5, 14], [46, 14], [47.5, 2.5], [64, 2.5], [66, 0.6], [80, 0.6],
    [80.3, 0.05], [82, 1.4], [91, 1.4], [95, 0.05], [200, 0.05]];
  const trail_memory = (b) => kf(tmKeys, b);
  const tgKeys = [[0, 0.55], [30, 0.55], [31.5, 1.0], [46, 1.0], [47.5, 0.55], [80, 0.8], [80.3, 0.0],
    [82, 0.45], [91, 0.45], [95, 0.0], [200, 0]];
  const trail_gain = (b) => kf(tgKeys, b);
  const zKeys = [[0, 1.45], [10, 1.36], [26, 1.0], [91, 1.0], [103, 0.9], [115, 0.86]];
  const zoom = (b) => kf(zKeys, b);
  const sunIKeys = [[0, 0], [91, 0], [93, 0.06], [99, 0.6], [105, 1.05], [115, 1.15]];
  const sunElKeys = [[0, 0.8], [91, 0.8], [105, 12.0], [115, 16.0]];
  function sun(b) {
    const u = clip01((b - 91) / 14.0);
    return {
      I: kf(sunIKeys, b), el: kf(sunElKeys, b),
      col: [1.0, 0.40 * (1 - u) + 0.74 * u, 0.36 * (1 - u) + 0.48 * u],
    };
  }
  const SUN_AZ = -24 * PI / 180;
  const fadeKeys = [[0, 0], [1.2, 0], [2.4, 1], [109, 1], [110.6, 0], [200, 0]];
  const fade = (b) => kf(fadeKeys, b);
  const juice = (t) => smooth((t - (ST.fall + 3.0)) / 5.0);

  return {
    FPS, SECTIONS, SB, ST, NBARS, TOTAL, t_of_B, B_of_t, smooth, kf, pos, angle, radius, intensity,
    light_height, ring_join, trail_memory, trail_gain, zoom, sun, SUN_AZ, fade, juice, ENTER, EXIT, PHI,
  };
})();
if (typeof module !== "undefined") module.exports = TL;
