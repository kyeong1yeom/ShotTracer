"""Golf ball tracer engine prototype (motion-detection based).
Same logic is ported to Kotlin (TracerEngine.kt)."""
import numpy as np
import cv2


class ImpactDetector:
    def __init__(self, ball_xy, ball_r, thr=18.0):
        self.cx, self.cy = int(ball_xy[0]), int(ball_xy[1])
        self.r = max(3, int(round(ball_r * 2)))
        self.thr = thr
        self.ref = None
        self.polarity = 1

    def roi(self, f):
        h, w = f.shape
        x0, x1 = max(0, self.cx - self.r), min(w, self.cx + self.r + 1)
        y0, y1 = max(0, self.cy - self.r), min(h, self.cy + self.r + 1)
        return f[y0:y1, x0:x1].astype(np.float32)

    def update(self, f):
        """returns True when ball region changed (impact)."""
        roi = self.roi(f)
        if self.ref is None:
            self.ref = roi
            c = float(f[self.cy, self.cx])
            self.polarity = 1 if c >= float(np.median(roi)) else -1
            return False
        d = float(np.mean(np.abs(roi - self.ref)))
        if d < self.thr * 0.3:  # slow adaptation to lighting
            self.ref = self.ref * 0.9 + roi * 0.1
        return d > self.thr


class BallTracker:
    def __init__(self, start_xy, ball_r, w, h, polarity=1, diff_thr=14, max_miss=8):
        self.start = np.array(start_xy, float)
        self.ball_r = ball_r
        self.w, self.h = w, h
        self.polarity = polarity
        self.diff_thr = diff_thr
        self.max_miss = max_miss
        self.frames, self.times = [], []
        self.pts = []             # (t, x, y)
        self.miss = 0
        self.done = False

    def velocity(self):
        """instantaneous velocity (px/s) at the newest detection, from a local fit"""
        q = np.array(self.pts[-4:])
        if len(q) < 2 or q[-1, 0] - q[0, 0] < 1e-6:
            return np.zeros(2)
        tt = q[:, 0] - q[-1, 0]
        return np.array([np.polyfit(tt, q[:, 1], 1)[0], np.polyfit(tt, q[:, 2], 1)[0]])

    def predict(self, t):
        """linear step with current velocity; ball decelerates on screen so the
        search window is centred a bit short of it (see push)"""
        last = np.array(self.pts[-1][1:])
        return last + self.velocity() * (t - self.pts[-1][0])

    MAX_GAP_SEC = 0.12
    MAX_BUF = 61

    def max_gap(self):
        if len(self.times) < 2:
            return 1
        fdt = max(self.times[-1] - self.times[-2], 1e-4)
        return int(max(1, min((self.MAX_BUF - 1) // 2, round(self.MAX_GAP_SEC / fdt))))

    def gap(self):
        """frame gap for differencing: ball must move ~ its own size between frames"""
        if len(self.pts) < 2 or len(self.times) < 2:
            return 1
        fdt = self.times[-1] - self.times[-2]
        v = self.speed() * fdt   # px / frame
        need = 2 * self.ball_r + 3
        return int(min(self.max_gap(), max(1, np.ceil(need / max(v, 0.05)))))

    def speed(self):
        return float(np.hypot(*self.velocity()))

    def push(self, f, t):
        if self.done:
            return
        self.frames.append(f.astype(np.int16)); self.times.append(t)
        if len(self.frames) > self.MAX_BUF:
            self.frames.pop(0); self.times.pop(0)
        g = self.gap()
        if len(self.frames) < 2 * g + 1:
            return
        a, b, c = self.frames[-1 - 2 * g], self.frames[-1 - g], self.frames[-1]
        tb = self.times[-1 - g]                  # detection is for middle frame
        if self.pts and tb <= self.pts[-1][0]:
            return
        if len(self.pts) < 2:   # search a big box above the tee / first detection
            ax, ay = self.start if not self.pts else self.pts[-1][1:]
            x0, x1 = ax - self.w * 0.3, ax + self.w * 0.3
            y0, y1 = ay - self.h * 0.6, ay
            pred = None
            if self.pts and tb - self.pts[-1][0] > 0.1:
                self.pts.clear()  # stale single detection -> restart
        else:
            last = np.array(self.pts[-1][1:])
            full = self.predict(tb)
            pred = last + 0.75 * (full - last)
            step = np.hypot(*(full - last))
            base = max(8.0, step * 0.45 + 6)
            rad = min(base + 1.0 * self.miss, 2 * base)
            x0, x1, y0, y1 = pred[0] - rad, pred[0] + rad, pred[1] - rad, pred[1] + rad
        x0, x1 = int(max(0, x0)), int(min(self.w, x1 + 1))
        y0, y1 = int(max(0, y0)), int(min(self.h, y1 + 1))
        if x1 <= x0 or y1 <= y0:
            return self._miss()
        db = (b - a)[y0:y1, x0:x1]
        mask = ((np.abs(db) > self.diff_thr) &
                (np.abs((c - b)[y0:y1, x0:x1]) > self.diff_thr)).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_area = max(30, (self.ball_r * 4) ** 2)
        best, bestd = None, 1e9
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                continue
            px, py = cents[i][0] + x0, cents[i][1] + y0
            if len(self.pts) < 3:
                if py > self.start[1] - self.ball_r * 1.5:
                    continue  # early detections must be above the tee
                if float(np.mean(db[labels == i])) * self.polarity <= 0:
                    continue  # club / body: wrong brightness polarity
            ref = (self.start if not self.pts else self.pts[-1][1:]) if pred is None else pred
            d = np.hypot(px - ref[0], py - ref[1])
            if pred is not None and d > rad:
                continue
            if d < bestd:
                best, bestd = (px, py), d
        if best is None:
            self._miss()
        else:
            self.pts.append((tb, best[0], best[1]))
            self.miss = 0

    MAX_MISS_SEC = 0.2

    def _miss(self, tb=None):
        if self.pts:
            self.miss += 1
            if self.times[-1] - self.pts[-1][0] > self.MAX_MISS_SEC + self.MAX_GAP_SEC:
                self.done = True


def _phi(t, k):
    return t if k < 1e-6 else (1.0 - np.exp(-k * t)) / k


def _fit_once(start_xy, pts, w, h, landing_fraction=0.3, t_max=8.0, dt=1 / 120):
    """Projective ballistic model with drag:
         u(t) = (A0 + A1*phi + A2*t^2) / (1 + a*phi),   v(t) = (B0 + B1*phi + B2*t^2) / (1 + a*phi)
         phi(t) = (1 - e^{-kt}) / k     (k = drag, grid searched; rest is linear least squares)
    """
    sx, sy = start_xy
    if len(pts) < 3:
        return [(sx, sy)] + [(p[1], p[2]) for p in pts]
    q = np.array(pts)
    k3 = min(2, len(q) - 1)
    step = np.hypot(q[k3, 1] - q[0, 1], q[k3, 2] - q[0, 2]) / max(1e-6, q[k3, 0] - q[0, 0])
    back = np.hypot(q[0, 1] - sx, q[0, 2] - sy) / max(step, 1e-6)
    t0 = q[0, 0] - min(max(back, 0.0), 0.25)
    t = np.concatenate([[0.0], q[:, 0] - t0])
    u = np.concatenate([[sx], q[:, 1]])
    v = np.concatenate([[sy], q[:, 2]])
    wt = np.ones_like(t); wt[0] = 4.0
    n = len(t)
    best = None
    if n >= 6:
        for k in np.concatenate([[0.0], np.geomspace(0.05, 4.0, 30)]):
            ph = _phi(t, k)
            A = np.zeros((2 * n, 7)); b = np.zeros(2 * n)
            A[0::2, 0], A[0::2, 1], A[0::2, 2], A[0::2, 6] = 1, ph, t * t, -ph * u
            A[1::2, 3], A[1::2, 4], A[1::2, 5], A[1::2, 6] = 1, ph, t * t, -ph * v
            b[0::2], b[1::2] = u, v
            Wt = np.repeat(wt, 2)
            sol = np.linalg.lstsq(A * Wt[:, None], b * Wt, rcond=None)[0]
            if sol[6] < -1e-3 * 0:  # depth must not shrink (ball never flies toward camera)
                pass
            D = 1 + sol[6] * ph
            res = np.sum(wt * ((sol[0] + sol[1] * ph + sol[2] * t * t) / D - u) ** 2 +
                         wt * ((sol[3] + sol[4] * ph + sol[5] * t * t) / D - v) ** 2)
            if np.all(D > 0.05) and (best is None or res < best[0]):
                best = (res, k, sol)
    if best is not None:
        _, k, s_ = best
        def f(tt):
            ph = _phi(tt, k)
            return (s_[0] + s_[1] * ph + s_[2] * tt * tt, s_[3] + s_[4] * ph + s_[5] * tt * tt, 1 + s_[6] * ph)
    else:
        cu = np.polyfit(t, u, 2, w=wt); cv = np.polyfit(t, v, 2, w=wt)
        f = lambda tt: (np.polyval(cu, tt), np.polyval(cv, tt), 1.0)

    out = []
    apex_v, apex_seen = None, False
    tt = 0.0
    while tt <= t_max:
        nu, nv, d = f(tt)
        if d < 0.05:
            break
        x, y = nu / d, nv / d
        if apex_v is None or y < apex_v:
            apex_v = y
        elif y > apex_v + 1:
            apex_seen = True
        if apex_seen and y >= apex_v + landing_fraction * (sy - apex_v):
            out.append((x, y)); break
        if not (-w < x < 2 * w and -h < y < 2 * h):
            break
        out.append((x, y))
        tt += dt
    return out


def fit_trajectory(start_xy, pts, w, h, **kw):
    """robust wrapper: drop detections far from a smooth local fit, then fit."""
    if len(pts) >= 7:
        q = np.array(pts)
        keep = np.ones(len(q), bool)
        for i in range(len(q)):
            lo, hi = max(0, i - 4), min(len(q), i + 5)
            idx = [j for j in range(lo, hi) if j != i]
            if len(idx) < 4:
                continue
            tt = q[idx, 0] - q[i, 0]
            px = np.polyval(np.polyfit(tt, q[idx, 1], 2), 0)
            py = np.polyval(np.polyfit(tt, q[idx, 2], 2), 0)
            if np.hypot(px - q[i, 1], py - q[i, 2]) > 6:
                keep[i] = False
        pts = [tuple(p) for p in q[keep]]
    return _fit_once(start_xy, pts, w, h, **kw)
