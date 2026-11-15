"""Renders the film, frame by frame, from the world model.

    python render.py test 20 55 90         # write test stills (seconds)
    python render.py all [workers]         # render the whole film to build/video.mp4

Top-down view. Four lanterns (the dancers) light the ground; the bush casts four
coloured shadows; feet leave dew prints, later mulberry prints.
"""
import math
import os
import pickle
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import timeline as TL

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
FF = os.path.join(HERE, "tools", "ffmpeg.exe")
W = H = 1080
SS = 2
TEXN = 1800              # ground texture covers world [-900, 900]

COL = np.array([[1.00, 0.30, 0.56],    # rose
                [0.22, 0.86, 0.74],    # teal
                [0.50, 0.42, 1.00],    # violet
                [1.00, 0.76, 0.30]])   # gold
AMB = np.array([0.006, 0.004, 0.016])
R_FALL = 165.0
FALLP = 1.3
EXPO = 4.8
GAM = 1 / 1.55
FONT_DIR = "C:/Windows/Fonts"

G = {}


def tm(x):
    return (1.0 - np.exp(-EXPO * np.maximum(x, 0.0))) ** GAM


# ------------------------------------------------------------------ setup
def make_ground():
    p = os.path.join(BUILD, "ground.png")
    if os.path.exists(p):
        return Image.open(p).convert("RGB")
    rng = np.random.default_rng(5)
    N = TEXN

    def noise(cell):
        n = N // cell + 3
        g = rng.random((n, n)).astype(np.float32)
        im = Image.fromarray(g, "F").resize((n * cell, n * cell), Image.BICUBIC)
        return np.asarray(im)[:N, :N]

    n1 = noise(300) * 0.5 + noise(120) * 0.3 + noise(45) * 0.2
    n1 = (n1 - n1.min()) / (n1.max() - n1.min())
    n2 = noise(16) * 0.6 + noise(6) * 0.4
    grain = rng.random((N, N)).astype(np.float32)
    m = np.clip((n1 - 0.25) / 0.5, 0, 1)
    m = m * m * (3 - 2 * m)
    soil = np.array([0.33, 0.27, 0.24])
    moss = np.array([0.27, 0.31, 0.21])
    n3 = noise(3) * 0.5 + noise(2) * 0.5
    col = soil * (1 - m)[..., None] + moss * m[..., None]
    col = col * (0.95 + 0.08 * n1)[..., None] * (0.86 + 0.22 * n2)[..., None] * (0.74 + 0.42 * n3)[..., None]         * (0.86 + 0.28 * grain)[..., None]
    img = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(90000):
        x, y = rng.uniform(0, N, 2)
        a = rng.uniform(0, 2 * math.pi)
        ln = rng.uniform(2, 7)
        if rng.random() < 0.5:
            c = (int(rng.uniform(90, 140)), int(rng.uniform(110, 160)), int(rng.uniform(60, 95)), int(rng.uniform(50, 130)))
        else:
            c = (int(rng.uniform(15, 35)), int(rng.uniform(15, 35)), int(rng.uniform(10, 25)), int(rng.uniform(60, 140)))
        d.line([(x, y), (x + ln * math.cos(a), y + ln * math.sin(a))], fill=c, width=1)
    for _ in range(900):
        x, y = rng.uniform(0, N, 2)
        r = rng.uniform(0.8, 2.2)
        v = int(rng.uniform(90, 130))
        d.ellipse([x - r, y - r * 0.8, x + r, y + r * 0.8], fill=(v, v - 8, v - 16, 140))
    img.save(p)
    return img


def leaf_outline():
    pts = [(0.00, 0.00), (0.06, 0.20), (0.22, 0.40), (0.45, 0.46), (0.68, 0.36), (0.86, 0.18),
           (1.00, 0.00), (0.86, -0.18), (0.68, -0.36), (0.45, -0.46), (0.22, -0.40), (0.06, -0.20)]
    return np.array(pts)


FS = 1.3
FOOT = np.array([(-10, 0), (-9.3, 2.6), (-7.5, 3.6), (-5, 3.6), (-2, 3.5), (1, 3.8), (4, 4.1), (6.2, 3.4),
                 (7, 1.0), (6.8, -1.5), (6, -3.6), (3.5, -4.2), (1, -3.2), (-1.5, -1.6), (-4.5, -1.8),
                 (-7.5, -3.0), (-9.3, -2.4)], float)
TOES = np.array([(9.1, -2.5, 1.75), (9.3, 0.3, 1.15), (8.7, 2.0, 1.0), (7.9, 3.4, 0.9), (7.0, 4.5, 0.8)], float)


def init():
    if G:
        return
    with open(os.path.join(BUILD, "world.pkl"), "rb") as f:
        G["w"] = pickle.load(f)
    G["ground"] = make_ground()
    G["leaf"] = leaf_outline()
    r = np.hypot(*np.meshgrid(np.arange(-140, 141), np.arange(-140, 141)))
    G["spr_r"] = r.astype(np.float32)
    G["font_ring"] = ImageFont.truetype(os.path.join(FONT_DIR, "GARA.TTF"), 31)
    G["font_title"] = ImageFont.truetype(os.path.join(FONT_DIR, "GARA.TTF"), 64)
    G["font_it"] = ImageFont.truetype(os.path.join(FONT_DIR, "GARAIT.TTF"), 34)
    G["font_small"] = ImageFont.truetype(os.path.join(FONT_DIR, "GARA.TTF"), 25)
    pxs = np.arange(W) + 0.5
    G["px"] = pxs
    bush = G["w"]["bush"]
    order = np.argsort(-bush["w"])
    G["seg_order"] = order


# --------------------------------------------------------------- lighting
def lights_at(t):
    b = float(TL.B_of_t(t))
    P = np.array([TL.pos(k, b) for k in range(4)])
    I = np.array([float(TL.intensity(k, b)) for k in range(4)])
    flick = np.array([1 + 0.035 * math.sin(2 * math.pi * 6.1 * t + 1.3 * k) + 0.025 * math.sin(2 * math.pi * 9.7 * t + 2.1 * k)
                      + 0.02 * math.sin(2 * math.pi * 2.3 * t + k) for k in range(4)])
    return b, P, I * flick, float(TL.light_height(b))


def spot(hx, hy, Lp):
    lr = math.hypot(Lp[0], Lp[1]) + 1e-6
    cx, cy = -Lp[0] / lr, -Lp[1] / lr
    hr = np.sqrt(hx * hx + hy * hy) + 40.0
    c = (hx * cx + hy * cy) / hr
    return 0.32 + 0.68 * ((1 + c) * 0.5) ** 1.6 * 1.25


def irr_points(x, y, z, P, I, Hl, sun):
    """Unshadowed irradiance at world points (N,) -> (N,3)."""
    out = np.tile(AMB, (len(x), 1))
    for k in range(4):
        if I[k] <= 1e-3:
            continue
        hx, hy = x - P[k, 0], y - P[k, 1]
        d2 = hx * hx + hy * hy + Hl ** 2
        E = I[k] * (Hl * Hl / d2) ** FALLP * spot(hx, hy, P[k])
        out += E[:, None] * COL[k]
    sI, sel, scol = sun
    if sI > 0:
        out += sI * (scol * 0.62 + np.array([0.08, 0.09, 0.15]) * 0.28)
    return out


# ------------------------------------------------------------ bush state
def seg_state(t):
    bush = G["w"]["bush"]
    S = bush["seg"]
    o = G["seg_order"]
    vis = o[S[o, 4] <= t]
    s = S[vis]
    frac = np.clip((t - s[:, 4]) / np.maximum(s[:, 5] - s[:, 4], 1e-3), 0, 1)
    x0, y0 = s[:, 0], s[:, 1]
    x1 = x0 + (s[:, 2] - x0) * frac
    y1 = y0 + (s[:, 3] - y0) * frac
    age = t - s[:, 4]
    w = bush["w"][vis] * (0.3 + 0.7 * TL.smooth(age / 22.0)) * np.minimum(1, 0.45 + age / 1.5)
    return x0, y0, bush["z0"][vis], x1, y1, bush["z1"][vis], w, age


def leaf_state(t):
    Lv = G["w"]["leaves"]
    if len(Lv) == 0:
        return None
    m = (Lv[:, 5] <= t)
    L = Lv[m]
    g = TL.smooth((t - L[:, 5]) / 3.5)
    return L, g


def leaf_poly(x, y, a, size, scale=1.0):
    o = G["leaf"]
    ca, sa = math.cos(a), math.sin(a)
    u = o[:, 0] * size * scale
    v = o[:, 1] * size * 0.78 * scale
    return x + u * ca - v * sa, y + u * sa + v * ca


def flying_leaf(L, t, b):
    """Leaves torn off by the whirl: spiral outward, counter-clockwise, rising."""
    s = t - L[7]
    r0 = math.hypot(L[0], L[1])
    th0 = math.atan2(L[1], L[0])
    r = r0 + 20 * s + 15 * s * s
    th = th0 + 1.05 * s + 0.32 * s * s
    rot = L[3] + L[13] * (2.6 * s + 0.5 * math.sin(3.1 * s + L[11]))
    sc = 1 + 0.8 * min(s, 1.6)
    return r * math.cos(th), r * math.sin(th), rot, sc


def berry_color(B, t):
    tflow, tset, tred, tblk = B[5], B[6], B[7], B[8]
    white = np.array([0.88, 0.88, 0.80])
    green = np.array([0.66, 0.80, 0.52])
    red = np.array([0.66, 0.035, 0.13])
    black = np.array([0.085, 0.010, 0.055])
    if t < tset:
        return green, 0.0
    u = TL.smooth((t - tset) / 4.0)
    c = green * (1 - u) + white * u
    u = TL.smooth((t - tred) / 3.0)
    c = c * (1 - u) + red * u
    u = TL.smooth((t - tblk) / 3.0)
    c = c * (1 - u) + black * u
    return c, float(u)


# -------------------------------------------------------------- drawing
def to2(x, y, S):
    return (W / 2 + S * x) * SS, (H / 2 - S * y) * SS


def shadow_masks(t, b, P, I, Hl, S, segst, leafst, sun):
    """Half-resolution shadow masks, one per lantern (+ one for the sun)."""
    x0, y0, z0, x1, y1, z1, w, age = segst
    HW = W // 2
    masks = []
    Bv = G["w"]["berries"]
    srcs = [(k, P[k]) for k in range(4) if I[k] > 0.02]
    for k, Lp in srcs:
        im = Image.new("L", (HW, HW), 0)
        d = ImageDraw.Draw(im)
        if len(x0):
            ze0 = np.minimum(z0, 0.8 * Hl)
            ze1 = np.minimum(z1, 0.8 * Hl)
            f0 = Hl / (Hl - ze0)
            f1 = Hl / (Hl - ze1)
            sx0 = Lp[0] + (x0 - Lp[0]) * f0
            sy0 = Lp[1] + (y0 - Lp[1]) * f0
            sx1 = Lp[0] + (x1 - Lp[0]) * f1
            sy1 = Lp[1] + (y1 - Lp[1]) * f1
            u0 = HW / 2 + 0.5 * S * sx0
            v0 = HW / 2 - 0.5 * S * sy0
            u1 = HW / 2 + 0.5 * S * sx1
            v1 = HW / 2 - 0.5 * S * sy1
            ww = w * (f0 + f1) * 0.25 * S * 1.5
            for i in range(len(u0)):
                wi = ww[i]
                if wi < 1.0:
                    d.line([(u0[i], v0[i]), (u1[i], v1[i])], fill=int(255 * max(wi, 0.45)), width=1)
                else:
                    d.line([(u0[i], v0[i]), (u1[i], v1[i])], fill=255, width=int(wi + 0.5))
        if leafst is not None:
            L, g = leafst
            for j in range(len(L)):
                if t >= L[j, 7] or g[j] < 0.05:
                    continue
                lx, ly = leaf_poly(L[j, 0], L[j, 1], L[j, 3], L[j, 4], g[j])
                ze = min(L[j, 2], 0.8 * Hl)
                f = Hl / (Hl - ze)
                sx = Lp[0] + (lx - Lp[0]) * f
                sy = Lp[1] + (ly - Lp[1]) * f
                d.polygon(list(zip(HW / 2 + 0.5 * S * sx, HW / 2 - 0.5 * S * sy)), fill=125)
        if len(Bv):
            for B in Bv:
                if t < B[5] or t > B[9] + 0.35:
                    continue
                ze = min(B[2], 0.8 * Hl)
                f = Hl / (Hl - ze)
                sx = Lp[0] + (B[0] - Lp[0]) * f
                sy = Lp[1] + (B[1] - Lp[1]) * f
                rr = 0.5 * S * f * 3.2 * B[4] * min(1, (t - B[5]) / 4)
                u, v = HW / 2 + 0.5 * S * sx, HW / 2 - 0.5 * S * sy
                d.ellipse([u - rr, v - rr, u + rr, v + rr], fill=230)
        im = im.filter(ImageFilter.GaussianBlur(1.3)).resize((W, H), Image.BILINEAR)
        masks.append((k, np.asarray(im, np.float32) / 255.0))
    sun_mask = None
    sI, sel, scol = sun
    if sI > 0.005 and len(x0):
        im = Image.new("L", (HW, HW), 0)
        d = ImageDraw.Draw(im)
        L = 1.0 / math.tan(math.radians(sel))
        dx, dy = math.cos(TL.SUN_AZ), math.sin(TL.SUN_AZ)
        sx0 = x0 - dx * z0 * L
        sy0 = y0 - dy * z0 * L
        sx1 = x1 - dx * z1 * L
        sy1 = y1 - dy * z1 * L
        u0 = HW / 2 + 0.5 * S * sx0
        v0 = HW / 2 - 0.5 * S * sy0
        u1 = HW / 2 + 0.5 * S * sx1
        v1 = HW / 2 - 0.5 * S * sy1
        ww = w * 0.5 * S
        for i in range(len(u0)):
            wi = ww[i]
            if wi < 1.0:
                d.line([(u0[i], v0[i]), (u1[i], v1[i])], fill=int(255 * max(wi, 0.3)), width=1)
            else:
                d.line([(u0[i], v0[i]), (u1[i], v1[i])], fill=255, width=int(wi + 0.5))
        blur = 1.2 + 2.5 * min(1.0, L / 30)
        im = im.filter(ImageFilter.GaussianBlur(blur)).resize((W, H), Image.BILINEAR)
        sun_mask = np.asarray(im, np.float32) / 255.0
    return masks, sun_mask


def draw_marks(t, S):
    """Footprints and berry stains, drawn into the ground's albedo."""
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im, "RGBA")
    Bv = G["w"]["berries"]
    stains = G["w"]["stains"]
    for i, B in enumerate(Bv):
        ts = B[9] + 0.35
        if t < ts:
            continue
        poly, drops = stains[i]
        fresh = math.exp(-(t - ts) / 6.0)
        c = (int(255 * (0.20 + 0.10 * fresh)), int(255 * 0.025), int(255 * (0.13 + 0.06 * fresh)), int(235))
        sc = min(1.0, (t - ts) / 0.12)
        cx, cy = B[0] + B[10], B[1] + B[11]
        pts = [to2(cx + px * sc, cy + py * sc, S) for px, py in poly]
        d.polygon(pts, fill=c)
        for dx, dy, r in drops:
            u, v = to2(cx + dx * sc, cy + dy * sc, S)
            rr = r * S * SS
            d.ellipse([u - rr, v - rr, u + rr, v + rr], fill=c)
    P = G["w"]["prints"]
    P = P[P[:, 0] <= t]
    for p in P:
        tp, x, y, a, foot, j, k = p
        age = t - tp
        dew = 0.0
        if age < 7.0:
            dew = 0.55 * (1 - age / 7.0) ** 1.3 * min(1, age / 0.12)
        alpha_j = 0.9 * j * min(1, age / 0.12) * (0.62 + 0.38 * math.exp(-age / 25.0)) * (0.7 + 0.3 * math.sin(tp * 37.1) ** 2)
        if dew < 0.01 and alpha_j < 0.01:
            continue
        fa, fb = math.cos(a), math.sin(a)
        ra, rb = math.sin(a), -math.cos(a)
        sgn = 1.0 if foot == 1 else -1.0
        u = FOOT[:, 0] * FS
        v = FOOT[:, 1] * sgn * FS
        wx = x + u * fa + v * ra
        wy = y + u * fb + v * rb
        pts = list(zip((W / 2 + S * wx) * SS, (H / 2 - S * wy) * SS))
        toes = []
        for tu, tv, tr in TOES * FS:
            tv *= sgn
            tx = x + tu * fa + tv * ra
            ty = y + tu * fb + tv * rb
            uu, vv = to2(tx, ty, S)
            rr = tr * S * SS
            toes.append([uu - rr, vv - rr, uu + rr, vv + rr])
        if dew >= 0.01:
            c = (160, 182, 200, int(255 * dew))
            d.polygon(pts, fill=c)
            for e in toes:
                d.ellipse(e, fill=c)
        if alpha_j >= 0.01:
            c = (52, 6, 38, int(255 * alpha_j))
            d.polygon(pts, fill=c)
            for e in toes:
                d.ellipse(e, fill=c)
    return im.reduce(SS)


def draw_bush(t, b, P, I, Hl, S, segst, leafst, sun):
    im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im, "RGBA")
    # the seed
    if t < TL.ST["grow"] + 6:
        a = 1 - TL.smooth((t - TL.ST["grow"] - 2) / 4)
        irr = irr_points(np.array([0.0]), np.array([0.0]), np.array([2.0]), P, I, Hl, sun)[0]
        c = tm(np.array([0.55, 0.20, 0.22]) * irr) * 255
        u, v = to2(0, 0, S)
        rx, ry = 4.2 * S * SS, 3.0 * S * SS
        d.ellipse([u - rx, v - ry, u + rx, v + ry], fill=(int(c[0]), int(c[1]), int(c[2]), int(255 * a)))
        hc = tm(irr * 0.9) * 255
        d.ellipse([u - rx * 0.5, v - ry * 0.7, u - rx * 0.05, v - ry * 0.15],
                  fill=(int(hc[0]), int(hc[1]), int(hc[2]), int(160 * a)))
    x0, y0, z0, x1, y1, z1, w, age = segst
    if len(x0):
        mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        irr = irr_points(mx, my, mz, P, I, Hl, sun)
        bark = np.array([0.33, 0.24, 0.21])
        young = np.array([0.62, 0.80, 0.42])
        yu = np.clip(1 - age / 3.0, 0, 1)[:, None]
        alb = bark * (1 - yu) + young * yu
        col = tm(alb * irr * 1.15) * 255
        u0 = (W / 2 + S * x0) * SS
        v0 = (H / 2 - S * y0) * SS
        u1 = (W / 2 + S * x1) * SS
        v1 = (H / 2 - S * y1) * SS
        ww = w * S * SS
        for i in range(len(u0)):
            wi = ww[i]
            c = col[i]
            if wi < 1.6:
                d.line([(u0[i], v0[i]), (u1[i], v1[i])], fill=(int(c[0]), int(c[1]), int(c[2]), int(255 * min(1, wi / 1.6))), width=1 if wi < 1.2 else 2)
            else:
                f = (int(c[0]), int(c[1]), int(c[2]), 255)
                d.line([(u0[i], v0[i]), (u1[i], v1[i])], fill=(12, 6, 10, 255), width=int(wi + 1.8))
                d.line([(u0[i], v0[i]), (u1[i], v1[i])], fill=f, width=int(wi + 0.5))
                if wi >= 3:
                    r = wi / 2
                    d.ellipse([u1[i] - r, v1[i] - r, u1[i] + r, v1[i] + r], fill=f)
    # leaves
    if leafst is not None:
        L, g = leafst
        tf = TL.ST["fall"]
        breeze = 0.06 + 0.25 * float(TL.smooth((t - tf) / 8.0))
        young = np.array([0.46, 0.68, 0.20])
        mature = np.array([0.12, 0.36, 0.12])
        att = (t < L[:, 7]) & (g > 0.03)
        if att.any():
            La = L[att]
            irr = irr_points(La[:, 0], La[:, 1], La[:, 2], P, I, Hl, sun)
            ga = g[att]
            for j in range(len(La)):
                lf = La[j]
                u = TL.smooth((t - lf[5]) / 9.0)
                alb = young * (1 - u) + mature * u
                ua = float(TL.smooth((t - lf[6]) / 3.0))
                alb = alb * (1 - ua) + lf[8:11] * ua
                c = tm(alb * irr[j] * (0.78 + 0.26 * lf[13]) + alb * 0.012) * 255
                wob = breeze * math.sin(1.7 * t + lf[11] * 3) * (1 + ua)
                lx, ly = leaf_poly(lf[0], lf[1], lf[3] + wob, lf[4], ga[j])
                pts = list(zip((W / 2 + S * lx) * SS, (H / 2 - S * ly) * SS))
                d.polygon(pts, fill=(int(c[0]), int(c[1]), int(c[2]), 245))
                if ga[j] > 0.4:
                    cc = (int(c[0] * 0.62), int(c[1] * 0.62), int(c[2] * 0.62), 200)
                    d.line([pts[0], pts[6]], fill=cc, width=1)
    # berries and flowers
    Bv = G["w"]["berries"]
    for B in Bv:
        if t < B[5] or t > B[9] + 0.35:
            continue
        irr = irr_points(np.array([B[0]]), np.array([B[1]]), np.array([B[2]]), P, I, Hl, sun)[0]
        u, v = to2(B[0], B[1], S)
        if t < B[6]:
            gfl = TL.smooth((t - B[5]) / 1.5) * (1 - TL.smooth((t - B[6] + 0.5) / 1.0))
            c = tm(np.array([0.85, 0.88, 0.70]) * irr * 1.3) * 255
            for q in range(3):
                aa = B[3] + q * 2.1
                rr = 1.3 * S * SS * gfl
                cx = u + math.cos(aa) * 2.2 * S * SS
                cy = v + math.sin(aa) * 2.2 * S * SS
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(int(c[0]), int(c[1]), int(c[2]), int(230 * gfl)))
            continue
        alb, ripe = berry_color(B, t)
        size = B[4] * min(1.0, 0.35 + 0.65 * (t - B[6]) / 5.0)
        if t > B[9]:
            size *= 1 - 0.55 * (t - B[9]) / 0.35
        c = tm(alb * irr * 1.2) * 255
        L = 4.6 * size * S * SS
        Wd = 3.0 * size * S * SS
        ca, sa = math.cos(B[3]), math.sin(B[3])
        # body as rotated ellipse (polygon)
        ang = np.linspace(0, 2 * math.pi, 14, endpoint=False)
        ex = L * np.cos(ang)
        ey = Wd * np.sin(ang)
        pts = list(zip(u + ex * ca - ey * sa, v + ex * sa + ey * ca))
        d.polygon(pts, fill=(int(c[0]), int(c[1]), int(c[2]), 255))
        # drupelets
        hl = tm(alb * irr * 1.2 + irr * 0.10 * (0.4 + ripe)) * 255
        for q, (du, dv) in enumerate(((-0.55, -0.35), (-0.1, 0.4), (0.35, -0.3), (0.7, 0.25), (-0.15, -0.05))):
            px_ = u + (du * L) * ca - (dv * Wd) * sa
            py_ = v + (du * L) * sa + (dv * Wd) * ca
            rr = Wd * 0.38
            d.ellipse([px_ - rr, py_ - rr, px_ + rr, py_ + rr], fill=(int(hl[0]), int(hl[1]), int(hl[2]), 150))
        sp = tm(irr * (0.5 + 0.8 * ripe)) * 255
        rr = max(1.0, Wd * 0.22)
        d.ellipse([u - L * 0.3 - rr, v - Wd * 0.3 - rr, u - L * 0.3 + rr, v - Wd * 0.3 + rr],
                  fill=(int(sp[0]), int(sp[1]), int(sp[2]), 210))
    # leaves in the whirlwind
    if leafst is not None:
        L, g = leafst
        fl = L[t >= L[:, 7]]
        for lf in fl:
            x, y, rot, sc = flying_leaf(lf, t, b)
            if abs(x) * S > W * 0.75 or abs(y) * S > H * 0.75:
                continue
            irr = irr_points(np.array([x]), np.array([y]), np.array([75.0]), P, I, Hl, sun)[0]
            c = tm(lf[8:11] * irr * (0.85 + 0.3 * lf[13])) * 255
            lx, ly = leaf_poly(x, y, rot, lf[4], sc)
            pts = list(zip((W / 2 + S * lx) * SS, (H / 2 - S * ly) * SS))
            d.polygon(pts, fill=(int(c[0]), int(c[1]), int(c[2]), 225))
            d.line([pts[0], pts[6]], fill=(int(c[0] * 0.6), int(c[1] * 0.6), int(c[2] * 0.6), 200), width=1)
    return im.reduce(SS)


def draw_emissive(t, b, P, I, Hl, S, sun):
    im = Image.new("RGB", (W * SS, H * SS), (0, 0, 0))
    d = ImageDraw.Draw(im)
    # light trails: silk threads spun by the lanterns
    Tm = float(TL.trail_memory(b))
    gain = float(TL.trail_gain(b))
    if gain > 0.01 and Tm > 0.06:
        n = int(min(900, max(12, Tm * 60)))
        ts = np.linspace(t - Tm, t, n)
        bs = TL.B_of_t(ts)
        age = t - ts
        fadek = np.exp(-age / max(Tm * 0.42, 0.05))
        for k in range(4):
            pk = TL.pos(k, bs)
            Ik = TL.intensity(k, bs)
            u = (W / 2 + S * pk[:, 0]) * SS
            v = (H / 2 - S * pk[:, 1]) * SS
            amp = gain * fadek * Ik * 0.75
            for i in range(n - 1):
                a = amp[i]
                if a < 0.02:
                    continue
                c = COL[k] * a * 255
                d.line([(u[i], v[i]), (u[i + 1], v[i + 1])], fill=(int(c[0]), int(c[1]), int(c[2])), width=2)
    # ring of joined hands
    J = float(TL.ring_join(b))
    if J > 0.01:
        th = np.arctan2(P[:, 1], P[:, 0])
        r = np.hypot(P[:, 0], P[:, 1])
        order = np.argsort(th)
        for q in range(4):
            i, j = order[q], order[(q + 1) % 4]
            a0, a1 = th[i], th[j]
            if a1 <= a0:
                a1 += 2 * math.pi
            m = 40
            uu = np.linspace(0, 1, m)
            aa = a0 + (a1 - a0) * uu
            rr = r[i] + (r[j] - r[i]) * uu - 10 * np.sin(math.pi * uu)
            xs = (W / 2 + S * rr * np.cos(aa)) * SS
            ys = (H / 2 - S * rr * np.sin(aa)) * SS
            amp = J * 0.55 * math.sqrt(max(I[i], 0) * max(I[j], 0))
            for s in range(m - 1):
                f = (s + 0.5) / (m - 1)
                c = (COL[i] * (1 - f) + COL[j] * f) * amp * 255
                d.line([(xs[s], ys[s]), (xs[s + 1], ys[s + 1])], fill=(int(c[0]), int(c[1]), int(c[2])), width=2)
    # dew glints in the first sun
    sI = sun[0]
    if sI > 0.02:
        Sp = G["w"]["sparkles"]
        tw = np.maximum(0, np.sin(Sp[:, 3] * t * 2.0 + Sp[:, 2])) ** 12 * Sp[:, 4] * min(1.0, sI * 1.3)
        sel = np.nonzero(tw > 0.04)[0]
        for i in sel:
            u, v = to2(Sp[i, 0], Sp[i, 1], S)
            a = tw[i]
            c = (int(255 * a), int(235 * a), int(200 * a))
            rr = 1.6
            d.ellipse([u - rr, v - rr, u + rr, v + rr], fill=c)
            d.line([(u - 5 * a - 2, v), (u + 5 * a + 2, v)], fill=c, width=1)
    em = np.asarray(im.reduce(SS), np.float32) / 255.0
    glow = np.asarray(Image.fromarray((np.clip(em, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)),
                      np.float32) / 255.0
    em = em + glow * 1.6
    # the lanterns themselves
    rgrid = G["spr_r"]
    for k in range(4):
        if I[k] < 0.01:
            continue
        cx = W / 2 + S * P[k, 0]
        cy = H / 2 - S * P[k, 1]
        sf = (Hl / 120.0) ** 0.6 * S ** 0.5
        rr = rgrid / sf
        core = np.exp(-(rr / 3.3) ** 2) * 1.25
        halo = np.exp(-(rr / 9.0) ** 2) * 0.75 + np.exp(-rr / 24.0) * 0.32
        add_sprite(em, cx, cy, core[..., None] * np.array([1, 0.97, 0.92]) + halo[..., None] * COL[k], I[k])
    # the seed calls the dancers in
    if t < TL.ST["grow"] + 4:
        a = 0.22 * (0.6 + 0.4 * math.sin(2 * math.pi * t / 3.0)) * (1 - TL.smooth((t - TL.ST["grow"]) / 4)) \
            * TL.smooth((t - 3.5) / 3.0)
        rr = rgrid / max(S, 0.5)
        spr = (np.exp(-(rr / 6.0) ** 2) * 0.8 + np.exp(-rr / 20.0) * 0.25)[..., None] * np.array([1.0, 0.75, 0.55])
        add_sprite(em, W / 2, H / 2, spr, a)
    return em


def add_sprite(em, cx, cy, spr, gain):
    h, w = spr.shape[:2]
    x0 = int(round(cx)) - w // 2
    y0 = int(round(cy)) - h // 2
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    em[ya:yb, xa:xb] += spr[ya - y0:yb - y0, xa - x0:xb - x0] * gain


# ------------------------------------------------------------ typography
RING_TEXT = "HERE WE GO ROUND THE MULBERRY BUSH  \u00b7  SO EARLY IN THE MORNING  \u00b7  "


def draw_ring_text(img, t, b, S):
    b0, b1 = 101.5, 107.5
    if b < b0:
        return
    t0, t1 = float(TL.t_of_B(b0)), float(TL.t_of_B(b1))
    font = G["font_ring"]
    chars = list(RING_TEXT)
    R = 470 * S
    adv = [font.getlength(c) + 7.0 for c in chars]
    total = sum(adv)
    scale = 2 * math.pi * R / total
    rot = 0.022 * (t - t0)
    end = float(TL.smooth((TL.TOTAL - 0.6 - t) / 1.2))
    acc = 0.0
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for i, c in enumerate(chars):
        a_mid = (acc + adv[i] / 2) * scale / R
        acc += adv[i]
        ti = t0 + (t1 - t0) * i / len(chars)
        al = float(TL.smooth((t - ti) / 0.8)) * end
        if al <= 0 or c == " ":
            continue
        th = math.pi / 2 - a_mid + rot
        gw = int(font.getlength(c)) + 12
        g = Image.new("L", (gw, 48), 0)
        ImageDraw.Draw(g).text((6, 4), c, font=font, fill=255)
        gr = g.rotate(math.degrees(th) - 90, resample=Image.BICUBIC, expand=True)
        cx = W / 2 + R * math.cos(th)
        cy = H / 2 - R * math.sin(th)
        col = Image.new("RGBA", gr.size, (246, 226, 236, 0))
        col.putalpha(gr.point(lambda v, al=al: int(v * al * 0.92)))
        layer.alpha_composite(col, (int(cx - gr.size[0] / 2), int(cy - gr.size[1] / 2)))
    img.alpha_composite(layer)


CREDITS = [
    ("title", "Mulberry Bush", 111.0),
    ("it", "a round for four dancers", 111.8),
    ("small", "after the 1976 film and dance by Ken Kahn", 112.8),
    ("small", "images and music written in code by Claude \u00b7 2026", 113.4),
]


def draw_credits(img, t, b):
    if b < 110.6:
        return
    d = ImageDraw.Draw(img, "RGBA")
    ys = {"title": 430, "it": 512, "small": 0}
    y_small = [610, 652]
    si = 0
    end = float(TL.smooth((TL.TOTAL - 0.6 - t) / 1.2))
    for kind, text, bb in CREDITS:
        al = float(TL.smooth((b - bb) / 0.6)) * end
        if kind == "title":
            f, y, c = G["font_title"], ys["title"], (240, 214, 230)
        elif kind == "it":
            f, y, c = G["font_it"], ys["it"], (205, 170, 215)
        else:
            f, y, c = G["font_small"], y_small[si], (170, 150, 175)
            si += 1
        if al <= 0:
            continue
        wdt = f.getlength(text)
        d.text(((W - wdt) / 2, y), text, font=f, fill=(c[0], c[1], c[2], int(255 * al)))


# ------------------------------------------------------------------ frame
def render(fi):
    init()
    t = fi / TL.FPS
    b, P, I, Hl = lights_at(t)
    S = float(TL.zoom(b))
    sun = TL.sun(b)
    segst = seg_state(t) if t >= TL.ST["grow"] else (np.zeros(0),) * 8
    leafst = leaf_state(t) if t >= TL.ST["leaf"] else None

    # albedo: ground texture under current zoom + footprints + stains
    c = TEXN / 2 - (W / 2) / S
    alb_im = G["ground"].transform((W, H), Image.AFFINE, (1 / S, 0, c, 0, 1 / S, c), resample=Image.BILINEAR)
    alb = np.asarray(alb_im, np.float32) / 255.0
    marks = np.asarray(draw_marks(t, S), np.float32) / 255.0
    ma = marks[..., 3:4]
    alb = alb * (1 - ma) + marks[..., :3] * ma

    # lighting
    X = (G["px"] - W / 2) / S
    Y = (H / 2 - G["px"]) / S
    X = X[None, :]
    Y = Y[:, None]
    masks, sun_mask = shadow_masks(t, b, P, I, Hl, S, segst, leafst, sun) if len(segst[0]) or leafst is not None \
        else ([], None)
    md = dict(masks)
    Lm = np.zeros((H, W, 3), np.float32) + AMB.astype(np.float32)
    R2 = R_FALL ** 2
    for k in range(4):
        if I[k] < 0.01:
            continue
        hx, hy = X - P[k, 0], Y - P[k, 1]
        d2 = hx * hx + hy * hy + Hl ** 2
        E = (I[k] * (Hl * Hl / d2) ** FALLP * spot(hx, hy, P[k])).astype(np.float32)
        if k in md:
            E = E * (1 - 0.93 * md[k])
        Lm += E[..., None] * COL[k].astype(np.float32)
    sI, sel, scol = sun
    if sI > 0:
        sl = sI * scol.astype(np.float32) * 0.62
        if sun_mask is not None:
            Lm += (1 - 0.88 * sun_mask)[..., None] * sl
        else:
            Lm += sl
        Lm += np.array([0.08, 0.09, 0.15], np.float32) * 0.28 * sI
    base = tm(alb * Lm)

    # the bush
    bush = np.asarray(draw_bush(t, b, P, I, Hl, S, segst, leafst, sun), np.float32) / 255.0
    ba = bush[..., 3:4]
    base = base * (1 - ba) + bush[..., :3] * ba

    # light
    em = draw_emissive(t, b, P, I, Hl, S, sun)
    base = 1 - (1 - base) * (1 - np.clip(em, 0, 1))

    # grade
    yy, xx = np.mgrid[0:H, 0:W]
    if "vig" not in G:
        rr = np.hypot(xx - W / 2, yy - H / 2) / (W * 0.72)
        G["vig"] = (1 - 0.42 * rr ** 2.4).astype(np.float32)[..., None]
    base = base * G["vig"]
    base = base * float(TL.fade(b))
    rng = np.random.default_rng(fi)
    base = base + rng.normal(0, 0.011, (H, W, 1)).astype(np.float32)
    out = Image.fromarray((np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8)).convert("RGBA")
    draw_ring_text(out, t, b, S)
    draw_credits(out, t, b)
    return np.asarray(out.convert("RGB"))


# ----------------------------------------------------------------- driver
def worker(args):
    wid, f0, f1 = args
    init()
    out = os.path.join(BUILD, f"chunk_{wid:02d}.mp4")
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(TL.FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
           "-pix_fmt", "yuv420p", "-x264-params", "keyint=60", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for fi in range(f0, f1):
        p.stdin.write(render(fi).tobytes())
        if (fi - f0) % 60 == 0:
            el = time.time() - t0
            print(f"[w{wid}] {fi - f0}/{f1 - f0} frames, {el:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        os.makedirs(os.path.join(BUILD, "test"), exist_ok=True)
        for s in sys.argv[2:]:
            t = float(s)
            t0 = time.time()
            a = render(int(round(t * TL.FPS)))
            Image.fromarray(a).save(os.path.join(BUILD, "test", f"t{t:06.1f}.png"))
            print(f"t={t}: {time.time() - t0:.2f}s")
    elif mode == "all":
        from multiprocessing import Pool
        nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        nf = int(math.floor(TL.TOTAL * TL.FPS))
        bounds = np.linspace(0, nf, nw + 1).astype(int)
        jobs = [(i, int(bounds[i]), int(bounds[i + 1])) for i in range(nw)]
        with Pool(nw) as pool:
            outs = pool.map(worker, jobs)
        lst = os.path.join(BUILD, "chunks.txt")
        with open(lst, "w") as f:
            for o in outs:
                f.write(f"file '{os.path.basename(o)}'\n")
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        os.path.join(BUILD, "video.mp4")], check=True)
        print("video done")


if __name__ == "__main__":
    main()
