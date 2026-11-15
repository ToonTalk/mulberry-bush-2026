"""The soundtrack: a four-voice round, one voice per lantern.

Each voice pans with its dancer as she circles, so you hear them go round.
When two dancers turn and go the other way, their voices sing the tune backwards.
Footsteps, berry plops, wind, crickets and birds come from the same world model.
Writes build/audio.wav
"""
import math
import os
import pickle
import wave

import numpy as np
from scipy.signal import oaconvolve, lfilter

import timeline as TL

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
SR = 48000
N = int((TL.TOTAL + 1.0) * SR)
rng = np.random.default_rng(76)

# the tune: (onset in eighths, duration in eighths, midi). 8 bars of 6/8, G major pentatonic
MEL = [(0, 1, 67), (1, 1, 67), (2, 1, 67), (3, 2, 67), (5, 1, 71),      # here we go round the
       (6, 1, 74), (7, 1, 74), (8, 1, 71), (9, 3, 67),                   # mul-ber-ry bush
       (12, 2, 69), (14, 1, 69), (15, 2, 76), (17, 1, 74),
       (18, 1, 71), (19, 1, 69), (20, 1, 67), (21, 3, 64),
       (24, 2, 74), (26, 1, 76), (27, 2, 79), (29, 1, 76),               # so ear-ly
       (30, 3, 74), (33, 3, 71),
       (36, 2, 69), (38, 1, 67), (39, 2, 64), (41, 1, 62),               # in the mor-
       (42, 3, 67)]                                                      # ning
RETRO = sorted([(48 - (o + d), d, p) for o, d, p in MEL])

VOICE_OCT = [0, 12, -12, 0]
VOICE_GAIN = [0.34, 0.20, 0.17, 0.30]


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def pan_gains(p):
    a = (np.clip(p, -1, 1) + 1) * math.pi / 4
    return math.cos(a), math.sin(a)


# ------------------------------------------------------------- instruments
def env_adsr(n, a, r, sus_len):
    e = np.ones(n)
    na = max(1, int(a * SR))
    e[:na] = np.linspace(0, 1, na)
    ns = int(sus_len * SR)
    nr = max(1, int(r * SR))
    if ns < n:
        tail = np.exp(-np.arange(n - ns) / nr * 3.0)
        e[ns:] *= tail
    return e


def pluck(f, dur):
    L = dur + 1.6
    t = np.arange(int(L * SR)) / SR
    out = np.zeros_like(t)
    for n in range(1, 16):
        fn = n * f * math.sqrt(1 + 0.0003 * n * n)
        if fn > 14000:
            break
        amp = abs(math.sin(math.pi * n * 0.17)) / n ** 1.05
        dec = 1.4 + 0.75 * n * (f / 440) ** 0.5
        out += amp * np.sin(2 * math.pi * fn * t + rng.uniform(0, 6.28)) * np.exp(-t * dec)
    att = np.minimum(1, t / 0.003)
    damp = np.where(t > dur + 0.35, np.exp(-(t - dur - 0.35) * 5), 1.0)
    return out * att * damp * 0.9


def flute(f, dur):
    L = dur + 0.3
    t = np.arange(int(L * SR)) / SR
    vib = 1 + 0.0045 * np.sin(2 * math.pi * 5.1 * t) * np.clip((t - 0.15) / 0.3, 0, 1)
    ph = 2 * math.pi * np.cumsum(f * vib) / SR
    s = np.sin(ph) + 0.20 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    br = lfilter([0.08], [1, -0.92], rng.normal(0, 1, len(t)))
    e = env_adsr(len(t), 0.07, 0.12, dur)
    breath_env = np.exp(-t / 0.12) * 0.6 + 0.25
    return (s + br * breath_env * 0.8) * e * 0.8


def reed(f, dur):
    L = dur + 0.25
    t = np.arange(int(L * SR)) / SR
    vib = 1 + 0.003 * np.sin(2 * math.pi * 4.4 * t) * np.clip((t - 0.2) / 0.4, 0, 1)
    ph = 2 * math.pi * np.cumsum(f * vib) / SR
    s = np.zeros_like(t)
    for n in range(1, 22):
        if n * f > 9000:
            break
        a = (1.0 if n % 2 else 0.32) / n
        a /= 1 + (n * f / 2200) ** 2
        s += a * np.sin(n * ph)
    e = env_adsr(len(t), 0.09, 0.1, dur)
    return s * e * 0.9


def marimba(f, dur):
    L = 2.2
    t = np.arange(int(L * SR)) / SR
    s = (np.sin(2 * math.pi * f * t) * np.exp(-t * (2.0 + f / 700))
         + 0.32 * np.sin(2 * math.pi * f * 3.98 * t) * np.exp(-t * 9)
         + 0.10 * np.sin(2 * math.pi * f * 9.9 * t) * np.exp(-t * 18)
         + 0.06 * np.sin(2 * math.pi * f * 2.0 * t) * np.exp(-t * 6))
    click = lfilter([0.3], [1, -0.6], rng.normal(0, 1, len(t))) * np.exp(-t / 0.004) * 0.25
    return (s + click) * np.minimum(1, t / 0.002)


INSTR = [pluck, flute, reed, marimba]


def bell(f, L=1.6):
    t = np.arange(int(L * SR)) / SR
    return (np.sin(2 * math.pi * f * t) * np.exp(-t * 3.5)
            + 0.4 * np.sin(2 * math.pi * f * 2.76 * t) * np.exp(-t * 7)
            + 0.2 * np.sin(2 * math.pi * f * 5.4 * t) * np.exp(-t * 12)) * np.minimum(1, t / 0.001)


def plop(f):
    L = 0.35
    t = np.arange(int(L * SR)) / SR
    fr = f * (1 + 1.4 * np.exp(-t / 0.018))
    ph = 2 * math.pi * np.cumsum(fr) / SR
    return np.sin(ph) * np.exp(-t / 0.07) * np.minimum(1, t / 0.0015)


def thump(bright):
    L = 0.16
    t = np.arange(int(L * SR)) / SR
    fr = 70 * (1 + 0.8 * np.exp(-t / 0.01))
    ph = 2 * math.pi * np.cumsum(fr) / SR
    body = np.sin(ph) * np.exp(-t / 0.035)
    nz = rng.normal(0, 1, len(t))
    nz = nz - lfilter([1 - 0.9], [1, -0.9], nz)          # crude high-pass
    brush = nz * np.exp(-t / (0.012 + 0.01 * bright)) * (0.25 + 0.3 * bright)
    return body + brush


def chirp_bird(base):
    syll = []
    n = rng.integers(2, 6)
    for _ in range(n):
        L = rng.uniform(0.04, 0.11)
        t = np.arange(int(L * SR)) / SR
        f0 = base * rng.uniform(0.85, 1.25)
        f1 = f0 * rng.uniform(0.6, 1.6)
        fr = f0 + (f1 - f0) * (t / L) + 300 * np.sin(2 * math.pi * rng.uniform(30, 70) * t)
        ph = 2 * math.pi * np.cumsum(fr) / SR
        e = np.sin(math.pi * t / L) ** 2
        syll.append(np.sin(ph) * e)
        syll.append(np.zeros(int(rng.uniform(0.02, 0.07) * SR)))
    return np.concatenate(syll)


def cricket():
    out = []
    for _ in range(3):
        L = 0.022
        t = np.arange(int(L * SR)) / SR
        out.append(np.sin(2 * math.pi * 4400 * t) * np.sin(math.pi * t / L) ** 2)
        out.append(np.zeros(int(0.018 * SR)))
    return np.concatenate(out)


# --------------------------------------------------------------- mixing bus
class Bus:
    def __init__(self):
        self.L = np.zeros(N)
        self.R = np.zeros(N)

    def add(self, sig, t, gain, pan=0.0):
        i0 = int(t * SR)
        if i0 >= N or i0 + len(sig) <= 0:
            return
        if i0 < 0:
            sig = sig[-i0:]
            i0 = 0
        n = min(len(sig), N - i0)
        gl, gr = pan_gains(pan)
        self.L[i0:i0 + n] += sig[:n] * gain * gl
        self.R[i0:i0 + n] += sig[:n] * gain * gr


def dyn(b):
    return float(TL.kf([(0, 0.8), (10, 0.9), (46, 0.85), (64, 0.9), (80, 1.2, "l"), (80.1, 0.75), (81, 0.75),
                        (91, 0.85), (105, 0.8), (115, 0.7)], b))


def dancer_pan(k, b):
    p = TL.pos(k, b)
    dist = float(np.hypot(*p))
    g = float(np.clip(1.25 - (dist - 380) / 500, 0.25, 1.0)) if dist > 380 else 1.0
    return float(np.clip(p[0] / 420, -1, 1) * 0.9), g


def play_events(bus, k, events, bar0, b_lo, b_hi, gain_extra=1.0, cut=None):
    for o, d, m in events:
        bo = bar0 + o / 6.0
        if not (b_lo <= bo < b_hi):
            continue
        t_on = float(TL.t_of_B(bo)) + rng.normal(0, 0.006)
        t_off = float(TL.t_of_B(bo + d / 6.0))
        dur = max(0.08, t_off - t_on) * 0.92
        sig = INSTR[k](mtof(m + VOICE_OCT[k]), dur)
        if cut is not None:
            tc = float(TL.t_of_B(cut))
            nkeep = int(max(0, tc - t_on) * SR)
            if nkeep < len(sig):
                fade = np.exp(-np.arange(len(sig) - nkeep) / (0.025 * SR))
                sig = sig.copy()
                sig[nkeep:] *= fade
        pan, g = dancer_pan(k, bo)
        acc = 1.15 if o % 6 == 0 else (1.0 if o % 3 == 0 else 0.85)
        bus.add(sig, t_on, VOICE_GAIN[k] * acc * g * dyn(bo) * gain_extra, pan)


def round_voice(bus, k, ref, b_lo, b_hi, retro=None, gain_extra=1.0, cut=None):
    c0 = int(math.floor((b_lo - ref) / 8)) - 1
    c1 = int(math.ceil((b_hi - ref) / 8)) + 1
    for c in range(c0, c1):
        start = ref + 8 * c
        if retro:
            r_lo, r_hi = retro
            play_events(bus, k, MEL, start, b_lo, min(b_hi, r_lo), gain_extra, cut)
            play_events(bus, k, RETRO, start, max(b_lo, r_lo), min(b_hi, r_hi), gain_extra, cut)
            play_events(bus, k, MEL, start, max(b_lo, r_hi), b_hi, gain_extra, cut)
        else:
            play_events(bus, k, MEL, start, b_lo, b_hi, gain_extra, cut)


def farewell(bus, k, bar):
    ev = [(o - 36, d, m) for o, d, m in MEL if o >= 36]
    play_events(bus, k, ev, bar, bar, bar + 2.01, 1.25)


# ------------------------------------------------------------------ drone
def drone(bus):
    segs = [(0, 46, [43, 50, 55]), (46, 64, [40, 47, 52]), (64, 80, [36, 43, 48, 55]),
            (81, 91, [40, 47, 52]), (91, 116, [43, 50, 55, 59])]
    amp_keys = [(0, 0), (0.6, 0), (3, 0.5), (10, 0.65), (46, 0.65), (64, 0.75), (79.9, 1.0, "l"),
                (80.04, 0), (81, 0), (83, 0.42), (91, 0.42), (93, 0.6), (109, 0.6), (114.5, 0)]
    t_all = np.arange(N) / SR
    b_all = TL.B_of_t(t_all)
    amp = TL.kf(amp_keys, b_all)
    xf = 0.75   # crossfade in bars
    L = np.zeros(N)
    R = np.zeros(N)
    for b0, b1, chord in segs:
        w = TL.smooth((b_all - (b0 - xf)) / (2 * xf)) * (1 - TL.smooth((b_all - (b1 - xf)) / (2 * xf)))
        if b0 == 80 or b1 == 80:
            pass
        idx = np.nonzero(w > 1e-4)[0]
        if len(idx) == 0:
            continue
        sl = slice(idx[0], idx[-1] + 1)
        t = t_all[sl]
        for m in chord:
            f = mtof(m)
            for det, pan in ((-0.004, -0.5), (0.004, 0.5)):
                ff = f * (1 + det)
                s = np.zeros(len(t))
                for n in range(1, 13):
                    if n * ff > 5000:
                        break
                    s += np.sin(2 * math.pi * n * ff * t + n * 1.3) / n ** 1.7
                lfo = 0.75 + 0.25 * np.sin(2 * math.pi * t / (7.0 + m % 5) + m)
                s *= lfo * w[sl] * amp[sl] * (0.05 / len(chord) ** 0.5) * (1.0 if m > 40 else 1.3)
                gl, gr = pan_gains(pan * 0.6)
                L[sl] += s * gl
                R[sl] += s * gr
    bus.L += L
    bus.R += R


# ---------------------------------------------------------------- textures
def noise_band(lo, hi, n):
    x = rng.normal(0, 1, n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n)


def textures(bus, world):
    t_all = np.arange(N) / SR
    b_all = TL.B_of_t(t_all)
    # rustle of unfurling leaves
    Lv = world["leaves"]
    hist, edges = np.histogram(Lv[:, 5], bins=int(TL.TOTAL * 10), range=(0, TL.TOTAL))
    dens = np.interp(t_all, edges[:-1] + 0.05, np.convolve(hist, np.ones(8) / 8, "same"))
    gx = np.arange(0, TL.TOTAL + 2, 0.12)
    gate = np.interp(t_all, gx, rng.random(len(gx))) ** 3
    rust = noise_band(2500, 8000, N) * np.clip(dens / 12, 0, 1) * 0.035 * (0.2 + 1.6 * gate)
    bus.L += rust * (0.8 + 0.2 * np.sin(t_all * 1.3))
    bus.R += rust * (0.8 + 0.2 * np.cos(t_all * 1.1))
    # the whirlwind
    det = Lv[:, 7]
    hist, edges = np.histogram(det, bins=int(TL.TOTAL * 10), range=(0, TL.TOTAL))
    ldens = np.interp(t_all, edges[:-1] + 0.05, np.convolve(hist, np.ones(10) / 10, "same"))
    wind_amp = TL.kf([(0, 0), (64, 0), (68, 0.25), (79.9, 1.0, "i"), (80.05, 0.0), (200, 0)], b_all)
    wn = noise_band(150, 1800, N)
    sweep = 0.6 + 0.4 * np.sin(2 * math.pi * np.cumsum(0.2 + 0.6 * wind_amp) / SR)
    wl = wn * wind_amp * sweep * 0.12
    bus.L += wl * (0.7 + 0.3 * np.sin(t_all * 2.1))
    bus.R += wl * (0.7 + 0.3 * np.cos(t_all * 2.1))
    gx2 = np.arange(0, TL.TOTAL + 2, 0.07)
    gate2 = np.interp(t_all, gx2, rng.random(len(gx2))) ** 4
    flutter = noise_band(1500, 7000, N) * np.clip(ldens / 15, 0, 1) * 0.04 * (0.15 + 2.0 * gate2)
    bus.L += flutter
    bus.R += np.roll(flutter, 2400)


def main():
    with open(os.path.join(BUILD, "world.pkl"), "rb") as f:
        world = pickle.load(f)
    bus = Bus()

    # ---- the round
    retro = (30.5, 46.0)
    round_voice(bus, 0, TL.ENTER[0], TL.ENTER[0], 80.0, cut=80.0)
    round_voice(bus, 1, TL.ENTER[1], TL.ENTER[1], 80.0, retro=retro, cut=80.0)
    round_voice(bus, 2, TL.ENTER[2], TL.ENTER[2], 80.0, cut=80.0)
    round_voice(bus, 3, TL.ENTER[3], TL.ENTER[3], 80.0, retro=retro, cut=80.0)
    # night: two voices, then one
    round_voice(bus, 1, TL.ENTER[1], 81.0, 91.0, gain_extra=0.9)
    round_voice(bus, 2, TL.ENTER[2], 81.0, 87.0, gain_extra=0.9)
    # dawn: the flute alone, from the top
    round_voice(bus, 1, 91.0, 91.0, 99.0, gain_extra=1.15)
    # each dancer sings the last phrase as she leaves
    for k, bar in TL.EXIT.items():
        farewell(bus, k, bar)
    # closing chord, one note for each of the four
    for i, (k, m) in enumerate([(2, 55), (3, 71), (0, 74), (1, 67)]):
        bo = 105.5 + i * 0.5
        sig = INSTR[k](mtof(m + VOICE_OCT[k] * 0), 3.5)
        bus.add(sig, float(TL.t_of_B(bo)), VOICE_GAIN[k] * 0.7, [-0.6, -0.2, 0.2, 0.6][i])
    bus.add(pluck(mtof(79), 2.0), float(TL.t_of_B(111.0)), 0.18, 0.15)
    bus.add(marimba(mtof(67), 2.0), float(TL.t_of_B(111.0)) + 0.01, 0.2, -0.15)

    # ---- footsteps
    for p in world["prints"]:
        tp, x, y, a, foot, j, k = p
        b = float(TL.B_of_t(tp))
        if 80 <= b < 81:
            continue
        g = 0.045 * (0.8 if 81 <= b < 91 else 1.0) * (0.7 + 0.5 * rng.random()) * (0.75 if 64 <= b < 80 else 1.0)
        bus.add(thump(min(1.0, j + 0.2)), tp + rng.normal(0, 0.004), g, float(np.clip(x / 420, -1, 1)) * 0.85)

    # ---- the bush: a music-box tink for each new shoot
    pent = [91, 93, 95, 98, 100, 103]
    for t, x, y, order in world["bush"]["tinks"]:
        if order < 1 or rng.random() < 0.45:
            continue
        m = pent[int(order + rng.integers(0, 3)) % len(pent)]
        bus.add(bell(mtof(m), 0.8), t, 0.018, float(np.clip(x / 300, -1, 1)) * 0.8)

    # ---- berries: glass for ripening, plops as they fall and burst
    Bv = world["berries"]
    for B in Bv:
        x = B[0]
        if rng.random() < 0.35:
            m = [86, 88, 91, 93, 95][rng.integers(0, 5)]
            bus.add(bell(mtof(m), 1.4), B[7] + 1.0, 0.025, float(np.clip(x / 300, -1, 1)) * 0.8)
        m = [55, 57, 59, 62, 64, 67][rng.integers(0, 6)]
        bus.add(plop(mtof(m + 12)), B[9] + 0.35, 0.10, float(np.clip(x / 300, -1, 1)) * 0.8)

    # ---- the hold: everyone falls down
    th = float(TL.t_of_B(80.12))
    t = np.arange(int(2.5 * SR)) / SR
    boom = np.sin(2 * math.pi * 46 * t * (1 + 0.5 * np.exp(-t / 0.05))) * np.exp(-t / 0.6)
    bus.add(boom, th, 0.45, 0.0)
    bus.add(thump(1.0) * 3, th, 0.25, -0.3)
    bus.add(thump(1.0) * 3, th + 0.03, 0.25, 0.3)

    # ---- night crickets, dawn birds
    t0, t1 = float(TL.t_of_B(81.5)), float(TL.t_of_B(97))
    for pan, period in ((-0.6, 0.83), (0.55, 1.07)):
        t = t0 + rng.uniform(0, 0.5)
        while t < t1:
            fade = min(1.0, (t - t0) / 3.0) * min(1.0, (t1 - t) / 6.0)
            bus.add(cricket(), t, 0.022 * fade, pan)
            t += period * rng.uniform(0.95, 1.05)
    tb0, tb1 = float(TL.t_of_B(92.5)), TL.TOTAL - 4
    t = tb0
    while t < tb1:
        u = (t - tb0) / (tb1 - tb0)
        bus.add(chirp_bird(rng.uniform(2600, 5200)), t, 0.035 * min(1, u * 3 + 0.3), rng.uniform(-0.9, 0.9))
        t += rng.uniform(0.25, 1.4) * (1.4 - 0.9 * min(1, u * 2))

    drone(bus)
    textures(bus, world)

    # ---- room
    Lr = int(3.2 * SR)
    t = np.arange(Lr) / SR
    irL = rng.normal(0, 1, Lr) * np.exp(-t / 0.75)
    irR = rng.normal(0, 1, Lr) * np.exp(-t / 0.75)
    irL = lfilter([0.35], [1, -0.65], irL)
    irR = lfilter([0.35], [1, -0.65], irR)
    irL[: int(0.012 * SR)] = 0
    irR[: int(0.017 * SR)] = 0
    irL /= np.sqrt((irL ** 2).sum())
    irR /= np.sqrt((irR ** 2).sum())
    wetL = oaconvolve(bus.L, irL)[:N]
    wetR = oaconvolve(bus.R, irR)[:N]
    L = bus.L + 0.42 * wetL
    R = bus.R + 0.42 * wetR
    # gentle fade at the very end
    tt = np.arange(N) / SR
    end = np.clip((TL.TOTAL - 0.2 - tt) / 2.5, 0, 1)
    L *= end
    R *= end
    pk = max(np.abs(L).max(), np.abs(R).max())
    L = np.tanh(L / pk * 1.15) / math.tanh(1.15) * 0.89
    R = np.tanh(R / pk * 1.15) / math.tanh(1.15) * 0.89
    st = np.stack([L, R], 1)
    out = os.path.join(BUILD, "audio.wav")
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())
    rms = np.sqrt((st ** 2).mean())
    print("wrote", out, "dur", round(N / SR, 2), "rms dBFS", round(20 * math.log10(rms), 1))


if __name__ == "__main__":
    main()
