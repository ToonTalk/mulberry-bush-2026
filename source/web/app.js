// The player: unpacks the data, starts the ground and sound workers, runs the
// clock and the controls. Picture time follows the audio clock while playing.
(async function main() {
  const $ = (id) => document.getElementById(id);
  const TOTAL = TL.TOTAL;
  const canvas = $("film"), errBox = $("err"), statusEl = $("status");
  const fail = (msg) => { errBox.textContent = msg; errBox.hidden = false; statusEl.textContent = ""; };
  const fmt = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60), r = Math.floor(s % 60); return m + ":" + String(r).padStart(2, "0"); };

  let world;
  try { world = await decodeData($("data").textContent); }
  catch (e) { fail("Could not unpack the film data (" + e.message + "). A current Chrome, Edge, Firefox or Safari is needed."); return; }
  try { Film.init(canvas, world); }
  catch (e) { fail("This browser could not start WebGL 2, which the film is drawn with. " + e.message); return; }

  // ---------------------------------------------------------- state and clock
  let t = 0, playing = false, dirty = true, clock0 = 0, t0 = 0, scrubbing = false, wasPlaying = false;
  let soundState = "preparing";              // preparing | ready | off
  // the picture follows the audio clock once the audio is really running; until then
  // (or if the browser never lets it start) it follows the wall clock
  let audioOn = false;
  const clock = () => {
    if (audioOn) { const s = Sound.time(); if (s !== null) return s; }
    return t0 + (performance.now() - clock0) / 1000;
  };
  function startClock(at) {
    t0 = at; clock0 = performance.now();
    Sound.stop();
    audioOn = soundState === "ready" && Sound.running && Sound.play(at);
  }
  let wake = null;
  async function keepAwake(on) {
    try {
      if (on && !wake && navigator.wakeLock) wake = await navigator.wakeLock.request("screen");
      if (!on && wake) { await wake.release(); wake = null; }
    } catch (e) { wake = null; }
  }
  function play() {
    if (soundState !== "off") Sound.unlock();      // inside the click: browsers allow sound from here
    if (t >= TOTAL - 0.05) t = 0;
    playing = true;
    startClock(t);
    document.body.classList.add("playing", "started");
    keepAwake(true);
    syncButtons();
  }
  function pause() {
    if (playing) t = Math.min(TOTAL, clock());
    playing = false;
    audioOn = false;
    Sound.stop();
    document.body.classList.remove("playing");
    keepAwake(false);
    dirty = true;
    syncButtons();
  }
  function seek(to) {
    t = Math.min(TOTAL, Math.max(0, to));
    if (playing) startClock(t);
    dirty = true;
  }

  // ---------------------------------------------------------- the strip of sections
  const strip = $("strip"), segWrap = $("segs"), head = $("playhead");
  const secs = TL.SECTIONS.map(([name], i) => {
    const a = TL.ST[name], b = i + 1 < TL.SECTIONS.length ? TL.ST[TL.SECTIONS[i + 1][0]] : TOTAL;
    return { name, a, b };
  });
  const LABEL = { dark: "dark", enter: "enter", grow: "grow", leaf: "leaf", fruit: "fruit", fall: "fall", hold: "fall down",
    night: "night", dawn: "morning", title: "title" };
  for (const s of secs) {
    const d = document.createElement("div");
    d.className = "seg";
    d.style.flexGrow = (s.b - s.a).toFixed(3);
    d.dataset.name = s.name;
    d.innerHTML = `<i></i><span>${LABEL[s.name]}</span>`;
    d.title = `${LABEL[s.name]} · ${fmt(s.a)}`;
    segWrap.appendChild(d);
    s.el = d;
  }
  // hide the labels of sections too narrow to hold them
  const fitLabels = () => secs.forEach((s) => {
    s.el.classList.remove("narrow");
    const sp = s.el.querySelector("span");
    s.el.classList.toggle("narrow", sp.scrollWidth > s.el.getBoundingClientRect().width + 1);
  });
  fitLabels();
  addEventListener("resize", fitLabels);
  document.fonts.ready.then(fitLabels);
  const tAt = (clientX) => {
    const r = segWrap.getBoundingClientRect();
    return Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * TOTAL;
  };
  strip.addEventListener("pointerdown", (e) => {
    scrubbing = true; wasPlaying = playing;
    if (playing) pause();
    strip.setPointerCapture(e.pointerId);
    seek(tAt(e.clientX));
  });
  strip.addEventListener("pointermove", (e) => { if (scrubbing) seek(tAt(e.clientX)); });
  const endScrub = () => { if (!scrubbing) return; scrubbing = false; if (wasPlaying) play(); };
  strip.addEventListener("pointerup", endScrub);
  strip.addEventListener("pointercancel", endScrub);
  strip.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { seek(t - 5); e.preventDefault(); }
    if (e.key === "ArrowRight") { seek(t + 5); e.preventDefault(); }
  });

  // ---------------------------------------------------------- buttons and keys
  const playBtn = $("play"), bigPlay = $("bigplay"), fsBtn = $("fs"), muteBtn = $("mute");
  function syncButtons() {
    playBtn.setAttribute("aria-label", playing ? "Pause" : "Play");
    playBtn.classList.toggle("is-playing", playing);
    bigPlay.querySelector(".lbl").textContent = soundState === "preparing" ? "play · the sound joins when ready"
      : (t > 0.1 ? "continue" : "play the film");
  }
  const toggle = () => (playing ? pause() : play());
  playBtn.addEventListener("click", toggle);
  bigPlay.addEventListener("click", play);
  canvas.addEventListener("click", () => { if (document.body.classList.contains("started")) toggle(); });
  // volume: the slider's position is squared into gain, so equal steps sound roughly even;
  // the level is remembered in this browser
  const volEl = $("vol");
  let volume = 1, muted = false;
  try { const v = parseFloat(localStorage.getItem("mulberry-volume")); if (v >= 0 && v <= 1) volume = v; } catch (e) { /* no storage */ }
  function applyVolume(save) {
    const silent = muted || volume === 0;
    Sound.setVolume(silent ? 0 : volume * volume);
    volEl.value = Math.round(volume * 100);
    volEl.style.setProperty("--v", (silent ? 0 : volume * 100).toFixed(1) + "%");
    volEl.setAttribute("aria-valuetext", silent ? "muted" : Math.round(volume * 100) + "%");
    muteBtn.classList.toggle("is-muted", silent);
    muteBtn.setAttribute("aria-label", silent ? "Unmute" : "Mute");
    if (save) { try { localStorage.setItem("mulberry-volume", String(volume)); } catch (e) { /* no storage */ } }
  }
  const nudgeVolume = (d) => { volume = Math.min(1, Math.max(0, Math.round((volume + d) * 20) / 20)); muted = false; applyVolume(true); };
  volEl.addEventListener("input", () => { volume = volEl.value / 100; muted = false; applyVolume(true); });
  muteBtn.addEventListener("click", () => {
    if (muted || volume === 0) { muted = false; if (volume === 0) volume = 0.6; }
    else muted = true;
    applyVolume(true);
  });
  applyVolume(false);
  const stage = $("stage");
  fsBtn.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await stage.requestFullscreen();
    } catch (e) { /* fullscreen not allowed here */ }
  });
  if (!document.fullscreenEnabled) fsBtn.hidden = true;
  document.addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("input, textarea")) return;
    if ((e.key === " " || e.key === "Enter") && e.target.closest && e.target.closest("button")) return;   // the button handles it
    if (e.key === " " || e.key === "k") { if (document.body.classList.contains("started") || !playing) toggle(); e.preventDefault(); }
    else if (e.key === "ArrowLeft" && e.target !== strip) seek(t - 5);
    else if (e.key === "ArrowRight" && e.target !== strip) seek(t + 5);
    else if (e.key === "Home") seek(0);
    else if (e.key === "f") fsBtn.click();
    else if (e.key === "m") muteBtn.click();
    else if (e.key === "ArrowUp") { nudgeVolume(0.1); e.preventDefault(); }
    else if (e.key === "ArrowDown") { nudgeVolume(-0.1); e.preventDefault(); }
  });

  // ---------------------------------------------------------- readouts
  const secName = $("secname");
  const tcur = $("tcur"), barno = $("barno"), dots = Array.from(document.querySelectorAll(".beats i")), perf = $("perf");
  $("ttot").textContent = fmt(TOTAL);
  let lastSec = "", frames = 0, msSum = 0, perfT = performance.now();
  function readouts(b) {
    tcur.textContent = fmt(t);
    head.style.left = (t / TOTAL * 100).toFixed(3) + "%";
    strip.setAttribute("aria-valuenow", Math.round(t));
    strip.setAttribute("aria-valuetext", fmt(t));
    const bi = Math.min(TL.NBARS - 1, Math.floor(b));
    barno.textContent = `bar ${bi + 1}`;
    const eighth = Math.min(5, Math.floor((b - bi) * 6));
    dots.forEach((d, i) => d.classList.toggle("on", i === eighth && t > 0));
    let cur = secs[0];
    for (const s of secs) if (t >= s.a) cur = s;
    if (cur.name !== lastSec) {
      secs.forEach((s) => s.el.classList.toggle("cur", s === cur));
      secName.textContent = LABEL[cur.name];
      lastSec = cur.name;
    }
  }

  // for checking frames against the rendered film
  window.__film = { render: (s) => { t = s; return Film.render(s); }, seek, tick: () => tick(true), get t() { return t; },
    get audioOn() { return audioOn; } };

  // ---------------------------------------------------------- loop
  function tick(once) {
    if (playing) {
      t = clock();
      if (t >= TOTAL) { t = TOTAL; pause(); }
      // audio became available (context resumed, or the sound finished preparing): hand over
      else if (!audioOn && soundState === "ready" && Sound.running) startClock(t);
    }
    if (playing || dirty) {
      const b = Film.render(t);
      readouts(b);
      dirty = false;
      frames++; msSum += Film.lastFrameMs;
      const now = performance.now();
      if (now - perfT > 1000) {
        perf.textContent = `${(msSum / frames).toFixed(1)} ms per frame`;
        frames = 0; msSum = 0; perfT = now;
      }
    }
    if (!once) requestAnimationFrame(() => tick());
  }
  requestAnimationFrame(() => tick());

  // ---------------------------------------------------------- ground, fonts, sound
  const blobWorker = (src) => new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
  try {
    const gw = blobWorker($("src-ground").textContent);
    gw.onmessage = (e) => {
      const { N, px } = e.data;
      Film.setGround(new ImageData(new Uint8ClampedArray(px.buffer), N, N));
      gw.terminate();
      dirty = true;
    };
    gw.postMessage({ N: Film.TEXN, seed: 20 });
  } catch (e) { /* the ground stays dark */ }
  Film.loadFonts().then(() => { dirty = true; });

  const LV = world.leaves;
  const born = new Float32Array(LV.rows), det = new Float32Array(LV.rows);
  for (let i = 0; i < LV.rows; i++) { born[i] = LV.a[i * LV.cols + 5]; det[i] = LV.a[i * LV.cols + 7]; }
  const payload = { events: world.events.a, syll: world.syll.a, gateRust: world.gate_rust.a, gateFlutter: world.gate_flutter.a,
    leafBorn: born, leafDet: det };
  const workerSrc = $("src-tl").textContent + "\n" + $("src-audio").textContent;
  // one sound worker per spare core; fewer on phones, where memory is tighter
  const phone = matchMedia("(pointer: coarse)").matches;
  const K = Math.max(2, Math.min(phone ? 3 : 6, (navigator.hardwareConcurrency || 4) - 1));
  const tStart = performance.now();
  syncButtons();
  try {
    await Sound.prepare(workerSrc, payload, K, (f) => {
      statusEl.textContent = `Composing the soundtrack in your browser · ${Math.round(f * 100)}%`;
    });
    soundState = "ready";
    statusEl.textContent = `Soundtrack ready · synthesised in ${((performance.now() - tStart) / 1000).toFixed(1)} s`;
  } catch (e) {
    soundState = "off";
    statusEl.textContent = "The sound could not be made in this browser, so the film will play silently.";
  }
  syncButtons();

})();
