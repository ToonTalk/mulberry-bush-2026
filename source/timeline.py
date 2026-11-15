"""Shared clock for picture and sound.

Everything in the film is driven by one musical clock measured in bars of 6/8.
Seconds <-> bars conversion handles the accelerando and the slower night/dawn.
World coordinates: origin = the bush, y up, units ~ screen pixels at zoom 1.
"""
import math
import numpy as np

FPS = 30
W = H = 1080

# name, bars, seconds-per-bar at start, at end
SECTIONS = [
    ("dark", 2, 1.5, 1.5),
    ("enter", 8, 1.5, 1.5),
    ("grow", 20, 1.5, 1.5),
    ("leaf", 16, 1.5, 1.5),
    ("fruit", 18, 1.5, 1.5),
    ("fall", 16, 1.5, 0.8),
    ("hold", 1, 2.2, 2.2),
    ("night", 10, 1.8, 1.8),
    ("dawn", 14, 1.8, 1.8),
    ("title", 10, 2.0, 2.0),
]

SB = {}
_b = 0
for _name, _n, _d0, _d1 in SECTIONS:
    SB[_name] = _b
    _b += _n
NBARS = _b

SUB = 480
_B = np.linspace(0.0, NBARS, NBARS * SUB + 1)
_dur = np.zeros_like(_B)
for _name, _n, _d0, _d1 in SECTIONS:
    _b0 = SB[_name]
    _m = (_B >= _b0) & (_B <= _b0 + _n)
    _u = (_B[_m] - _b0) / _n
    _dur[_m] = _d0 + (_d1 - _d0) * _u
_T = np.concatenate([[0.0], np.cumsum((_dur[:-1] + _dur[1:]) * 0.5 * np.diff(_B))])
TOTAL = float(_T[-1])


def t_of_B(b):
    return np.interp(b, _B, _T)


def B_of_t(t):
    return np.interp(t, _T, _B)


ST = {k: float(t_of_B(v)) for k, v in SB.items()}


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def kf(keys, b):
    """Keyframe interpolation. keys: (bar, value[, ease]) ease in s/l/i/o."""
    b = np.asarray(b, float)
    scalar = b.ndim == 0
    b = np.atleast_1d(b)
    xs = np.array([k[0] for k in keys], float)
    vs = np.array([k[1] for k in keys], float)
    es = np.array([k[2] if len(k) > 2 else "s" for k in keys])
    i = np.clip(np.searchsorted(xs, b, side="right") - 1, 0, len(xs) - 2)
    x0, x1 = xs[i], xs[i + 1]
    u = np.clip((b - x0) / np.maximum(x1 - x0, 1e-9), 0, 1)
    e = es[i]
    u = np.where(e == "s", u * u * (3 - 2 * u),
        np.where(e == "i", u * u,
        np.where(e == "o", 1 - (1 - u) ** 2, u)))
    if vs.ndim == 1:
        out = vs[i] + (vs[i + 1] - vs[i]) * u
    else:
        out = vs[i] + (vs[i + 1] - vs[i]) * u[:, None]
    return out[0] if scalar else out


# ---------------------------------------------------------------- dancers
NDANCER = 4
R0 = 345.0
PHI = [math.pi / 4 + k * math.pi / 2 for k in range(4)]   # corners they enter from
ENTER = [2.0, 4.0, 6.0, 8.0]                              # bar each dancer starts walking in
EXIT = {0: 97.0, 3: 99.0, 2: 101.0, 1: 103.0}             # bar each dancer starts walking out
CORNER_R = 880.0

_RHO_EVEN = [(0, 0.0), (10.4, 0.0), (12.2, 1 / 8), (46, 1 / 8), (48, 1 / 16), (64, 1 / 16, "i"), (80, 0.30, "l"),
             (80.22, 0.0), (81.6, 0.0), (83.2, 1 / 16), (91, 1 / 16), (95, 0.0)]
_RHO_ODD = [(0, 0.0), (10.4, 0.0), (12.2, 1 / 8), (30, 1 / 8), (32, -1 / 8), (46, -1 / 8), (48, 1 / 16), (64, 1 / 16, "i"),
            (80, 0.30, "l"), (80.22, 0.0), (81.6, 0.0), (83.2, 1 / 16), (91, 1 / 16), (95, 0.0)]


def _integrate(rho):
    return np.concatenate([[0.0], np.cumsum((rho[:-1] + rho[1:]) * 0.5 * np.diff(_B))])


_ANG = []
for _k in range(4):
    _rho = kf(_RHO_EVEN if _k % 2 == 0 else _RHO_ODD, _B)
    _ANG.append(PHI[_k] + 2 * math.pi * _integrate(_rho))
# align the stopping positions at dawn with the four diagonals, so each dancer
# can walk straight back out through the corner it came in by
_i95 = np.searchsorted(_B, 95.0)
_a95 = _ANG[0][_i95] - PHI[0]
_delta = -(((_a95 + math.pi / 4) % (math.pi / 2)) - math.pi / 4)
_corr = _delta * smooth((_B - 88.0) / 7.0)
_ANG = np.array([a + _corr for a in _ANG])

# radius: base ring, the weave (leaf), breathing in/out (fruit), centrifugal (fall)
_base = kf([(0, R0), (64, R0), (66, R0 + 15), (80, R0 + 35, "l"), (80.22, R0 + 35), (81.5, R0 + 10),
            (83.5, R0), (200, R0)], _B)
_RAD = np.array([_base.copy() for _ in range(4)])
_wv = kf([(0, 0), (30.5, 0), (32.5, 1), (45, 1), (46.5, 0)], _B)
_A_WEAVE = 50.0
for _a in (0, 2):
    for _bb in (1, 3):
        _s = 1.0 if (_a + _bb) % 4 == 1 else -1.0
        _D = _ANG[_a] - _ANG[_bb]
        _x = _D - 2 * math.pi * np.round(_D / (2 * math.pi))
        _bump = np.exp(-(_x / 0.6) ** 2) * _wv
        _RAD[_a] += _A_WEAVE * _s * _bump
        _RAD[_bb] -= _A_WEAVE * _s * _bump
_io_all = kf([(0, 0), (46.5, 0), (48, 1), (55, 1), (56, 0), (200, 0)], _B)
_io_alt = kf([(0, 0), (55, 0), (56, 1), (63, 1), (64.5, 0), (200, 0)], _B)
for _k in range(4):
    _RAD[_k] -= 78 * _io_all * (0.5 - 0.5 * np.cos(2 * math.pi * (_B - 48) / 4))
    _RAD[_k] -= 78 * _io_alt * (0.5 - 0.5 * np.cos(2 * math.pi * (_B - 56) / 4 + math.pi * (_k % 2)))


def angle(k, b):
    return np.interp(b, _B, _ANG[k])


def radius(k, b):
    return np.interp(b, _B, _RAD[k])


def pos(k, b):
    """World position of dancer k at bar b (vectorised). Returns (..., 2)."""
    b = np.asarray(b, float)
    th = angle(k, b)
    r = radius(k, b)
    p = np.stack([r * np.cos(th), r * np.sin(th)], -1)
    c_in = CORNER_R * np.array([math.cos(PHI[k]), math.sin(PHI[k])])
    e = smooth((b - ENTER[k]) / 3.2)[..., None]
    p = c_in + (p - c_in) * e
    th_out = float(angle(k, EXIT[k]))
    c_out = CORNER_R * np.array([math.cos(th_out), math.sin(th_out)])
    x = np.clip((b - EXIT[k]) / 3.4, 0, 1)
    e2 = (x * x * (3 - 2 * x) * 0.5 + x * x * 0.5)[..., None]
    return p + (c_out - p) * e2


def intensity(k, b):
    b = np.asarray(b, float)
    base = kf([(0, 1.0), (79, 1.12), (80.15, 0.30), (81.2, 0.30), (83, 0.62), (91, 0.62),
               (99, 0.48), (200, 0.48)], b)
    leave = 1 - smooth((b - EXIT[k] - 0.5) / 3.0)
    return base * leave


def light_height(b):
    return kf([(0, 150), (79.8, 155), (80.4, 62), (81.6, 62), (83.5, 135), (91, 135), (97, 150), (200, 150)], b)


def ring_join(b):
    return kf([(0, 0), (10.5, 0), (12.5, 1), (30, 1), (30.7, 0), (46.8, 0), (48.6, 0.75), (64, 0.75),
               (66, 1.0), (80, 1.0), (80.12, 0.0), (200, 0)], b)


def trail_memory(b):
    return kf([(0, 0.7), (30, 0.7), (31.5, 14), (46, 14), (47.5, 2.5), (64, 2.5), (66, 0.6), (80, 0.6),
               (80.3, 0.05), (82, 1.4), (91, 1.4), (95, 0.05), (200, 0.05)], b)


def trail_gain(b):
    return kf([(0, 0.55), (30, 0.55), (31.5, 1.0), (46, 1.0), (47.5, 0.55), (80, 0.8), (80.3, 0.0),
               (82, 0.45), (91, 0.45), (95, 0.0), (200, 0)], b)


def zoom(b):
    return kf([(0, 1.45), (10, 1.36), (26, 1.0), (91, 1.0), (103, 0.9), (115, 0.86)], b)


def sun(b):
    """intensity, elevation(deg), colour"""
    I = kf([(0, 0), (91, 0), (93, 0.06), (99, 0.6), (105, 1.05), (115, 1.15)], b)
    el = kf([(0, 0.8), (91, 0.8), (105, 12.0), (115, 16.0)], b)
    u = float(np.clip((b - 91) / 14.0, 0, 1))
    col = np.array([1.0, 0.40, 0.36]) * (1 - u) + np.array([1.0, 0.74, 0.48]) * u
    return float(I), float(el), col


SUN_AZ = math.radians(-24)   # sun sits low at the bottom-right; shadows run up-left


def fade(b):
    return kf([(0, 0), (1.2, 0), (2.4, 1), (109, 1), (110.6, 0), (200, 0)], b)


def juice(t):
    """How much mulberry juice is on the dancers' feet."""
    return float(smooth((t - (ST["fall"] + 3.0)) / 5.0))


if __name__ == "__main__":
    print("bars", NBARS, "total", round(TOTAL, 2))
    for k, v in SB.items():
        print(f"{k:6s} bar {v:4d}  t {ST[k]:7.2f}")
