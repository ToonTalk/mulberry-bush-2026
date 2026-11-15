"""Packs everything the browser version needs into build/web/data.b64.

    python web_export.py

1. The world (bush segments in render order, leaves, berries, stains, prints,
   sparkles), quantised to int16 per column.
2. The score: every sound event audio.py places, with its exact time, gain,
   pan and parameters. audio.py is run with its instruments wrapped so that the
   random number stream is consumed exactly as in the real render; the events
   are recorded instead of mixed. The browser then only has to synthesise the
   timbres, so it plays the same notes, footsteps, tinks and birds.
"""
import base64
import gzip
import json
import math
import os
import pickle
import struct

import numpy as np

import timeline as TL

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
OUT = os.path.join(BUILD, "web")

KIND = {"pluck": 0, "flute": 1, "reed": 2, "marimba": 3, "bell": 4, "plop": 5, "thump": 6,
        "cricket": 7, "chirp": 8, "boom": 9}


# ------------------------------------------------------------------ score
def record_score():
    import audio as A

    events = []      # kind, t, gain, pan, p1, p2, cut_t, scale
    syll = []        # L, f0, f1, vib, gap
    gates = {}
    last = {}

    def wrap(name, fn):
        def w(*a):
            arr = fn(*a)
            last.clear()
            last.update(kind=name, args=a, arr=arr)
            return arr
        return w

    orig = dict(pluck=A.pluck, flute=A.flute, reed=A.reed, marimba=A.marimba, bell=A.bell,
                plop=A.plop, thump=A.thump, cricket=A.cricket)
    W = {k: wrap(k, v) for k, v in orig.items()}
    A.INSTR[:] = [W["pluck"], W["flute"], W["reed"], W["marimba"]]
    for k, v in W.items():
        setattr(A, k, v)

    def chirp_bird(base):
        # identical random draws to audio.chirp_bird, recorded instead of synthesised
        i0 = len(syll)
        n = A.rng.integers(2, 6)
        total = 0
        for _ in range(n):
            L = A.rng.uniform(0.04, 0.11)
            f0 = base * A.rng.uniform(0.85, 1.25)
            f1 = f0 * A.rng.uniform(0.6, 1.6)
            vib = A.rng.uniform(30, 70)
            gap = A.rng.uniform(0.02, 0.07)
            syll.append((L, f0, f1, vib, gap))
            total += int(L * A.SR) + int(gap * A.SR)
        arr = np.zeros(total)
        last.clear()
        last.update(kind="chirp", args=(i0, n), arr=arr)
        return arr

    A.chirp_bird = chirp_bird

    def add(self, sig, t, gain, pan=0.0):
        if last and len(sig) == len(last["arr"]):
            ref = last["arr"]
            scale = 1.0
            if sig is not ref:
                j = int(np.argmax(np.abs(ref)))
                scale = float(sig[j] / ref[j]) if ref[j] != 0 else 1.0
            kind, args = last["kind"], last["args"]
        else:
            kind, args, scale = "boom", (), 1.0
        p = list(args) + [0.0, 0.0]
        events.append((KIND[kind], t, gain, pan, p[0], p[1], -1.0, scale))
        last.clear()

    A.Bus.add = add

    def play_events(bus, k, evs, bar0, b_lo, b_hi, gain_extra=1.0, cut=None):
        # copy of audio.play_events that records the cut time instead of applying it
        for o, d, m in evs:
            bo = bar0 + o / 6.0
            if not (b_lo <= bo < b_hi):
                continue
            t_on = float(TL.t_of_B(bo)) + A.rng.normal(0, 0.006)
            t_off = float(TL.t_of_B(bo + d / 6.0))
            dur = max(0.08, t_off - t_on) * 0.92
            f = A.mtof(m + A.VOICE_OCT[k])
            A.INSTR[k](f, dur)
            last.clear()
            pan, g = A.dancer_pan(k, bo)
            acc = 1.15 if o % 6 == 0 else (1.0 if o % 3 == 0 else 0.85)
            tc = float(TL.t_of_B(cut)) if cut is not None else -1.0
            events.append((k, t_on, A.VOICE_GAIN[k] * acc * g * A.dyn(bo) * gain_extra, pan, f, dur, tc, 1.0))

    A.play_events = play_events
    A.drone = lambda bus: None      # deterministic; the browser computes it

    class Stop(Exception):
        pass

    def textures(bus, world):
        # same random draws as audio.textures, keeping only the gate values
        gx = np.arange(0, TL.TOTAL + 2, 0.12)
        gates["rust"] = A.rng.random(len(gx))
        A.rng.normal(0, 1, A.N)          # noise_band(2500, 8000)
        A.rng.normal(0, 1, A.N)          # noise_band(150, 1800)
        gx2 = np.arange(0, TL.TOTAL + 2, 0.07)
        gates["flutter"] = A.rng.random(len(gx2))
        raise Stop

    A.textures = textures
    A.Bus.__init__ = lambda self: None
    try:
        A.main()
    except Stop:
        pass
    return np.array(events, np.float64), np.array(syll, np.float64), gates


# ------------------------------------------------------------------ packing
class Packer:
    def __init__(self):
        self.buf = bytearray()
        self.man = {}

    def _put(self, name, raw, meta):
        while len(self.buf) % 4:
            self.buf.append(0)
        meta["off"] = len(self.buf)
        self.buf += raw
        self.man[name] = meta

    def q16(self, name, a):
        a = np.atleast_2d(np.asarray(a, np.float64))
        mx = np.abs(a).max(0)
        sc = np.where(mx > 0, 32760.0 / np.maximum(mx, 1e-12), 1.0)
        q = np.round(a * sc).astype("<i2")
        self._put(name, q.tobytes(), dict(t="q16", rows=a.shape[0], cols=a.shape[1], sc=sc.tolist()))

    def f32(self, name, a):
        a = np.atleast_2d(np.asarray(a, np.float64))
        self._put(name, a.astype("<f4").tobytes(), dict(t="f32", rows=a.shape[0], cols=a.shape[1]))

    def u8(self, name, a):
        a = np.asarray(a, np.float64).ravel()
        self._put(name, np.round(np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes(),
                  dict(t="u8", rows=len(a), cols=1))

    def blob(self):
        head = json.dumps(self.man).encode()
        return struct.pack("<I", len(head)) + head + bytes(self.buf)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(BUILD, "world.pkl"), "rb") as f:
        w = pickle.load(f)
    bush = w["bush"]
    order = np.argsort(-bush["w"])            # the order render.py draws them in
    S = bush["seg"][order]
    seg = np.column_stack([S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], S[:, 5],
                           bush["w"][order], bush["z0"][order], bush["z1"][order]])
    pk = Packer()
    pk.q16("seg", seg)
    pk.q16("leaves", w["leaves"])
    pk.q16("berries", w["berries"][:, :12])
    st = []
    for poly, drops in w["stains"]:
        drops = list(drops) + [(0.0, 0.0, 0.0)] * (6 - len(drops))     # radius 0 = no drop
        st.append([c for p in poly for c in p] + [c for d in drops for c in d])
    pk.q16("stains", np.array(st))
    pk.q16("prints", w["prints"])
    pk.q16("sparkles", w["sparkles"])

    ev, syll, gates = record_score()
    pk.f32("events", ev)
    pk.f32("syll", syll)
    pk.u8("gate_rust", gates["rust"])
    pk.u8("gate_flutter", gates["flutter"])

    raw = pk.blob()
    z = gzip.compress(raw, 9)
    b64 = base64.b64encode(z).decode()
    with open(os.path.join(OUT, "data.b64"), "w") as f:
        f.write(b64)
    kinds = {v: k for k, v in KIND.items()}
    cnt = {}
    for e in ev:
        cnt[kinds[int(e[0])]] = cnt.get(kinds[int(e[0])], 0) + 1
    print("events", len(ev), cnt, "syllables", len(syll))
    print("raw", len(raw), "gzip", len(z), "base64", len(b64))


if __name__ == "__main__":
    main()
