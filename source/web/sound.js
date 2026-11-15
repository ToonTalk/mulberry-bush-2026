// Main-thread side of the soundtrack: runs the synthesis workers (each adds the
// room to its window, as audio.py's oaconvolve), sums the windows, applies the
// end fade and the soft limiter, then plays it in sync with the picture.
const Sound = (() => {
  let buffer = null, ctx = null, src = null, gainNode = null;
  let startCtx = 0, startOff = 0;
  let volume = 1;
  const timing = {};

  // window boundaries with roughly equal work in each: every second costs the
  // drone and the room; each event costs its length times its instrument's rate
  // (rates in ns per sample, measured)
  function splitByWork(ev, T, K) {
    const BIN = 0.25, nb = Math.ceil((T + 1) / BIN);
    const cost = new Float64Array(nb).fill(BIN * 48000 * 480);
    const rate = [63, 60, 100, 16, 11, 20, 67, 20, 40, 40];
    const len = (k, p2) => [p2 + 1.6, p2 + 0.3, p2 + 0.25, 2.2, p2, 0.35, 0.16, 0.12, 0.3, 2.5][k];
    for (let o = 0; o < ev.length; o += 8) {
      const k = Math.round(ev[o]);
      cost[Math.min(nb - 1, Math.floor(ev[o + 1] / BIN))] += len(k, ev[o + 5]) * 48000 * rate[k];
    }
    const total = cost.reduce((a, b) => a + b, 0);
    const out = [0];
    let acc = 0;
    for (let i = 0; i < nb && out.length < K; i++) {
      acc += cost[i];
      if (acc >= total * out.length / K) out.push((i + 1) * BIN);
    }
    while (out.length < K) out.push(T);
    return out;
  }

  // K workers each synthesise one window of the timeline and add the room to it;
  // the windows (and their reverb tails) are summed here
  function prepare(workerSrc, payload, K, onProgress) {
    const SR = 48000, N = Math.floor((TL.TOTAL + 1.0) * SR), T = TL.TOTAL + 1.0;
    const url = URL.createObjectURL(new Blob([workerSrc], { type: "text/javascript" }));
    const prog = new Array(K).fill(0);
    const L = new Float32Array(N), R = new Float32Array(N);
    const t0 = performance.now();
    const bounds = splitByWork(payload.events, T, K);
    timing.windows = bounds.map((v) => +v.toFixed(1));
    const jobs = [];
    for (let wi = 0; wi < K; wi++) {
      const win = [bounds[wi], wi === K - 1 ? T + 1 : bounds[wi + 1]];
      jobs.push(new Promise((resolve, reject) => {
        const w = new Worker(url);
        w.onerror = (e) => { w.terminate(); reject(new Error(e.message || "sound worker failed")); };
        w.onmessage = (e) => {
          const d = e.data;
          if (!d.done) {
            prog[wi] = d.progress;
            onProgress(0.95 * prog.reduce((a, b) => a + b, 0) / K);
            return;
          }
          w.terminate();
          prog[wi] = 1;
          (timing.workers = timing.workers || [])[wi] = d.ms;
          const m = Math.min(d.L.length, N - d.s0);
          for (let i = 0; i < m; i++) { L[d.s0 + i] += d.L[i]; R[d.s0 + i] += d.R[i]; }
          resolve();
        };
        w.postMessage({ data: payload, win, wi });
      }));
    }
    return Promise.all(jobs).then(() => {
      URL.revokeObjectURL(url);
      timing.synth = performance.now() - t0;
      buffer = finish(N, L, R);
      onProgress(1);
      return buffer;
    });
  }

  // end fade and the soft limiter of audio.py, then an AudioBuffer to play
  function finish(N, L, R) {
    const SR = 48000, total = TL.TOTAL;
    let pk = 0;
    for (let i = 0; i < N; i++) {
      const e = Math.min(1, Math.max(0, (total - 0.2 - i / SR) / 2.5));
      L[i] *= e; R[i] *= e;
      const a = Math.abs(L[i]), b = Math.abs(R[i]);
      if (a > pk) pk = a;
      if (b > pk) pk = b;
    }
    const k = 1.15 / pk, nrm = 0.89 / Math.tanh(1.15);
    for (let i = 0; i < N; i++) { L[i] = Math.tanh(L[i] * k) * nrm; R[i] = Math.tanh(R[i] * k) * nrm; }
    const buf = new AudioBuffer({ length: N, numberOfChannels: 2, sampleRate: SR });
    buf.copyToChannel(L, 0);
    buf.copyToChannel(R, 1);
    return buf;
  }

  // must be called from inside a click or key press
  function unlock() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      ctx = new AC({ latencyHint: "playback" });
      gainNode = ctx.createGain();
      gainNode.gain.value = volume;
      gainNode.connect(ctx.destination);
    }
    if (ctx.state === "suspended") ctx.resume();
  }
  function heard() {
    if (ctx.getOutputTimestamp) {
      const ts = ctx.getOutputTimestamp();
      if (ts && ts.contextTime > 0 && ts.performanceTime > 0) return ts.contextTime + (performance.now() - ts.performanceTime) / 1000;
    }
    return ctx.currentTime - (ctx.outputLatency || ctx.baseLatency || 0);
  }
  function play(offset) {
    if (!buffer || !ctx) return false;
    stop();
    src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(gainNode);
    startCtx = ctx.currentTime + 0.03;
    startOff = offset;
    src.start(startCtx, Math.max(0, offset));
    return true;
  }
  function stop() {
    if (src) { try { src.stop(); } catch (e) { /* already stopped */ } src.disconnect(); src = null; }
  }
  function time() {
    if (!src) return null;
    return startOff + Math.max(0, heard() - startCtx);
  }
  function setVolume(v) { volume = v; if (gainNode) gainNode.gain.setTargetAtTime(v, ctx.currentTime, 0.02); }
  return { timing, prepare, unlock, play, stop, time, setVolume, get ready() { return !!buffer; }, get active() { return !!src; },
    get running() { return !!(buffer && ctx && ctx.state === "running"); } };
})();
