"""Precomputes everything that exists in the world: the bush (grown by the dance),
its leaves and berries, every footprint, berry stain and dew sparkle.
Writes build/world.pkl, used by both render.py and audio.py.
"""
import math
import os
import pickle

import numpy as np

import timeline as TL

OUT = os.path.join(os.path.dirname(__file__), "build", "world.pkl")


def zfun(x, y):
    # vase-shaped shrub: stems rise from the base at the centre to a canopy at the rim
    r = np.hypot(x, y)
    return 4.0 + 66.0 * np.clip(r / 270.0, 0, 1.1) ** 0.65


# --------------------------------------------------------------------- bush
def grow_bush(rng):
    t0 = TL.ST["grow"]
    t1 = TL.ST["leaf"] + 20.0
    dt = 1 / 30
    V0, TAU, K_TURN, K_CURL, SIG, MAXO, MAXT = 16.0, 23.0, 1.0, 0.26, 0.42, 5, 380
    CELL = 4.0
    G = int(1400 / CELL)
    grid = -np.ones((G, G), np.int32)

    tips = []
    segs = []      # x0 y0 x1 y1 ts te order parentseg tip
    tinks = []     # (t, x, y) when a new shoot starts (audio)

    def new_tip(x, y, a, order, parent, born, ptip=-1):
        if ptip >= 0:
            tips[ptip]["kin"].add(len(tips))
        tips.append(dict(x=x, y=y, a=a, order=order, last=parent, born=born, sx=x, sy=y, st=born,
                         kin={ptip} if ptip >= 0 else set(),
                         since=0.0, nxt=rng.uniform(15, 28) * 0.9 ** order, alive=True, id=len(tips),
                         plen=0.0, side=1 if rng.random() < 0.5 else -1))
        tinks.append((born, x, y, order))

    def close(tp, t):
        if math.hypot(tp["x"] - tp["sx"], tp["y"] - tp["sy"]) > 0.25:
            segs.append([tp["sx"], tp["sy"], tp["x"], tp["y"], tp["st"], t, tp["order"], tp["last"], tp["id"]])
            tp["last"] = len(segs) - 1
        tp["sx"], tp["sy"], tp["st"] = tp["x"], tp["y"], t

    base = rng.uniform(0, 2 * math.pi)
    n0 = 7
    for k in range(n0):
        a = base + 2 * math.pi * k / n0 + rng.uniform(-0.2, 0.2)
        new_tip(1.5 * math.cos(a), 1.5 * math.sin(a), a, 0, -1, t0 + 0.45 * k)

    t = t0
    while t < t1:
        b = TL.B_of_t(t)
        L = np.array([TL.pos(k, b) for k in range(4)])
        I = np.array([TL.intensity(k, b) for k in range(4)])
        n_alive = sum(1 for tp in tips if tp["alive"])
        i = 0
        while i < len(tips):
            tp = tips[i]
            i += 1
            if not tp["alive"] or t < tp["born"]:
                continue
            age = t - tp["born"]
            v = V0 * math.exp(-(t - t0) / TAU) * (0.83 ** tp["order"]) * min(1.0, 0.25 + age / 1.5)
            if v < 0.55:
                close(tp, t)
                tp["alive"] = False
                continue
            px, py = tp["x"], tp["y"]
            dx = L[:, 0] - px
            dy = L[:, 1] - py
            d = np.hypot(dx, dy) + 1e-6
            w = I / (1 + (d / 170.0) ** 2)
            tx = float((w * dx / d).sum())
            ty = float((w * dy / d).sum())
            r = math.hypot(px, py) + 1e-6
            tx += 0.55 * px / r
            ty += 0.55 * py / r
            da = (math.atan2(ty, tx) - tp["a"] + math.pi) % (2 * math.pi) - math.pi
            tp["a"] += K_TURN * da * dt + K_CURL * dt + rng.normal(0, SIG) * math.sqrt(dt)
            tp["x"] += v * math.cos(tp["a"]) * dt
            tp["y"] += v * math.sin(tp["a"]) * dt
            tp["plen"] += v * dt
            tp["since"] += v * dt
            x, y = tp["x"], tp["y"]
            th = math.atan2(y, x)
            rmax = 252 + 22 * math.sin(3 * th + 1.1) + 12 * math.sin(5 * th + 0.3) + 7 * math.sin(7 * th + 2.0)
            if math.hypot(x, y) > rmax:
                close(tp, t)
                tp["alive"] = False
                continue
            ci, cj = int((x + 700) / CELL), int((y + 700) / CELL)
            o = grid[ci, cj]
            if o != -1 and o != tp["id"] and o not in tp["kin"] and tp["plen"] > 10:
                close(tp, t)
                tp["alive"] = False
                continue
            grid[ci, cj] = tp["id"]
            if math.hypot(x - tp["sx"], y - tp["sy"]) >= 4.5:
                close(tp, t)
            if tp["since"] >= tp["nxt"] and tp["order"] < MAXO and n_alive < MAXT:
                close(tp, t)
                tp["side"] = -tp["side"]
                ca = tp["a"] + tp["side"] * rng.uniform(0.45, 0.85)
                new_tip(x, y, ca, tp["order"] + 1, tp["last"], t, tp["id"])
                n_alive += 1
                tp["a"] -= tp["side"] * rng.uniform(0.06, 0.2)
                tp["since"] = 0.0
                tp["nxt"] = rng.uniform(15, 28) * 0.9 ** tp["order"]
        t += dt
    for tp in tips:
        if tp["alive"]:
            close(tp, t)

    S = np.array(segs, float)
    n = len(S)
    parent = S[:, 7].astype(int)
    nchild = np.zeros(n, int)
    for i in range(n):
        if parent[i] >= 0:
            nchild[parent[i]] += 1
    leafc = (nchild == 0).astype(float)
    for i in range(n - 1, -1, -1):
        if parent[i] >= 0:
            leafc[parent[i]] += leafc[i]
    wfin = 0.7 + 0.82 * np.sqrt(leafc)
    z0 = zfun(S[:, 0], S[:, 1])
    z1 = zfun(S[:, 2], S[:, 3])
    return dict(seg=S, w=wfin, z0=z0, z1=z1, leafc=leafc, tinks=np.array(tinks)), tips


# -------------------------------------------------------------------- leaves
def make_leaves(bush, rng):
    S = bush["seg"]
    acc = {}
    side = {}
    out = []
    tl = TL.ST["leaf"]
    tf = TL.ST["fall"]
    for i, s in enumerate(S):
        x0, y0, x1, y1, ts, te, order, par, tip = s
        tip = int(tip)
        L = math.hypot(x1 - x0, y1 - y0)
        acc[tip] = acc.get(tip, rng.uniform(0, 8)) + L
        sp = 12.5 if order >= 1 else 15.0
        r = math.hypot(x1, y1)
        if acc[tip] >= sp and r > 38:
            acc[tip] = 0.0
            side[tip] = -side.get(tip, 1)
            a = math.atan2(y1 - y0, x1 - x0) + side[tip] * rng.uniform(0.65, 1.15)
            size = rng.uniform(9.5, 15.0) * (1.0 - 0.06 * order)
            born = max(te + 1.2, tl + 0.6 + (r / 260.0) * 10.0 + rng.uniform(0, 3.5))
            turn = tf + rng.uniform(0.0, 4.5)
            det = tf + 3.0 + (1 - r / 270.0) * 5.0 + rng.uniform(0, 9.5)
            fallc = [(0.92, 0.76, 0.18), (0.95, 0.62, 0.12), (0.86, 0.42, 0.12), (0.62, 0.16, 0.28)][
                rng.choice(4, p=[0.45, 0.28, 0.17, 0.10])]
            out.append([x1, y1, float(zfun(x1, y1)) + 3.0, a, size, born, turn, det,
                        *fallc, rng.uniform(0, 2 * math.pi), rng.uniform(0.8, 1.25),
                        rng.uniform(-1, 1)])
    return np.array(out, float)


# ------------------------------------------------------------------- berries
def make_berries(bush, rng):
    S = bush["seg"]
    leafc = bush["leafc"]
    acc = {}
    out = []
    tfr = TL.ST["fruit"]
    tf = TL.ST["fall"]
    for i, s in enumerate(S):
        x0, y0, x1, y1, ts, te, order, par, tip = s
        tip = int(tip)
        L = math.hypot(x1 - x0, y1 - y0)
        acc[tip] = acc.get(tip, rng.uniform(0, 18)) + L
        r = math.hypot(x1, y1)
        if order >= 2 and leafc[i] <= 3 and r > 55 and acc[tip] >= 34:
            acc[tip] = 0.0
            ang = math.atan2(y1 - y0, x1 - x0) + math.pi / 2 * (1 if rng.random() < 0.5 else -1)
            off = rng.uniform(2.5, 5.5)
            bx, by = x1 + off * math.cos(ang), y1 + off * math.sin(ang)
            tflow = tfr + 0.5 + rng.uniform(0, 4.5)
            tset = tflow + 3.5 + rng.uniform(0, 1.5)
            tred = tfr + 11.0 + (r / 260.0) * 4.0 + rng.uniform(0, 3.5)
            tblk = tred + 5.0 + rng.uniform(0, 2.5)
            tdrop = tf + 2.0 + rng.uniform(0, 14.5)
            out.append([bx, by, float(zfun(bx, by)) + 1.0, rng.uniform(0, math.pi), rng.uniform(0.85, 1.2),
                        tflow, tset, tred, tblk, tdrop, rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5),
                        rng.integers(0, 1 << 30)])
    return np.array(out, float)


def stain_shape(seed):
    rng = np.random.default_rng(int(seed))
    n = 11
    R = rng.uniform(3.4, 5.4)
    poly = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = R * rng.uniform(0.75, 1.25)
        poly.append((rr * math.cos(a), rr * math.sin(a)))
    drops = []
    for _ in range(rng.integers(3, 7)):
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(R + 1.5, R + 7.5)
        drops.append((d * math.cos(a), d * math.sin(a), rng.uniform(0.6, 1.9)))
    return poly, drops


# --------------------------------------------------------------- footprints
def step_pattern(bar):
    """Eighth-note offsets within a 6/8 bar and which foot lands (0 = left, 1 = right)."""
    if TL.SB["fall"] <= bar < TL.SB["hold"]:
        return [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1)]          # running
    if bar >= TL.SB["hold"]:
        return [(0, 0), (3, 1)]                                          # walking
    return [(0, 0), (2, 0), (3, 1), (5, 1)]                              # 6/8 skip: step-hop, step-hop


def make_footprints():
    prints = []   # t, x, y, angle, foot, juice, dancer
    for k in range(4):
        exit_steps = 0
        for bar in range(TL.NBARS):
            for off, foot in step_pattern(bar):
                b = bar + off / 6.0
                p = TL.pos(k, b)
                v = (TL.pos(k, b + 0.03) - TL.pos(k, b - 0.03)) / 0.06
                sp = float(np.hypot(*v))
                lim = 545 / float(TL.zoom(b)) + 25
                if abs(p[0]) > lim or abs(p[1]) > lim or sp < 12:
                    continue
                a = math.atan2(v[1], v[0])
                rx, ry = math.sin(a), -math.cos(a)
                lat = 8.5 if foot == 1 else -8.5
                cx, cy = p[0] + rx * lat, p[1] + ry * lat
                fa = a + (-0.13 if foot == 1 else 0.13)
                t = float(TL.t_of_B(b))
                j = TL.juice(t)
                if b >= TL.EXIT[k]:
                    exit_steps += 1
                    j *= math.exp(-exit_steps / 3.0)
                prints.append([t, cx, cy, fa, foot, j, k])
    P = np.array(prints, float)
    return P[np.argsort(P[:, 0])]


def main():
    rng = np.random.default_rng(1976)
    bush, _ = grow_bush(rng)
    leaves = make_leaves(bush, rng)
    berries = make_berries(bush, rng)
    prints = make_footprints()
    stains = [stain_shape(s) for s in berries[:, 12]]
    sp = rng.uniform(-1, 1, (1600, 2)) * 640
    sparkles = np.column_stack([sp, rng.uniform(0, 2 * math.pi, 1600), rng.uniform(0.6, 2.2, 1600),
                                rng.uniform(0.3, 1.0, 1600)])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(dict(bush=bush, leaves=leaves, berries=berries, prints=prints, stains=stains,
                         sparkles=sparkles), f)
    print("segments", len(bush["seg"]), "leaves", len(leaves), "berries", len(berries),
          "prints", len(prints), "max width", bush["w"].max().round(2))


if __name__ == "__main__":
    main()
