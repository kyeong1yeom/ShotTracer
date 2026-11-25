"""Golf ball tracer engine prototype (motion-detection based).
Same logic is ported to Kotlin (TracerEngine.kt)."""
import numpy as np
import cv2


def refine_ball(f, tap, w):
    """Snap a finger tap to the actual ball: find the compact bright (or dark) blob near the tap.
    returns (x, y, r, polarity) — falls back to the tap itself."""
    h = f.shape[0]
    R = int(max(10, 0.04 * w))
    tx, ty = int(round(tap[0])), int(round(tap[1]))
    x0, x1, y0, y1 = max(0, tx - R), min(w, tx + R + 1), max(0, ty - R), min(h, ty + R + 1)
    win = f[y0:y1, x0:x1].astype(np.float32)
    med = float(np.median(win))
    best = None
    for pol in (1, -1):
        c = (win - med) * pol
        peak = float(c.max())
        if peak < 20:
            continue
        mask = (c > 0.5 * peak).astype(np.uint8)
        n, lab, st, ce = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if area > (0.04 * w) ** 2 * 3.2 or max(bw, bh) > 2.5 * min(bw, bh) + 2:
                continue  # too big / elongated (club shaft, body)
            cx, cy = ce[i][0] + x0, ce[i][1] + y0
            d = np.hypot(cx - tap[0], cy - tap[1])
            strength = float(c[lab == i].mean())
            score = strength - 1.5 * d - (25 if pol < 0 else 0)   # white balls are the norm
            if best is None or score > best[0]:
                best = (score, cx, cy, max(1.5, np.sqrt(area / np.pi) * 1.1), pol)
    if best is None:
        return float(tap[0]), float(tap[1]), max(2.0, w * 0.01), 1
    return best[1], best[2], best[3], best[4]


class ImpactDetector:
    """Impact = the BALL ITSELF disappears from the tee and stays gone.
    A club moving next to the ball (address, waggle, backswing) does not trigger it."""

    def __init__(self, ball_xy, ball_r, polarity=1, fps_hint=None):
        self.cx, self.cy = float(ball_xy[0]), float(ball_xy[1])
        self.r = ball_r
        self.pol = polarity
        self.c0 = None            # ball contrast vs. surroundings when armed
        self.run_start = None     # time the ball was first seen missing
        self.run_n = 0
        self.impact_time = None

    def _contrast(self, f):
        h, w = f.shape
        rc = max(1.0, self.r * 0.7)
        R = int(np.ceil(self.r * 3.2)) + 1
        x0, x1 = int(max(0, self.cx - R)), int(min(w, self.cx + R + 1))
        y0, y1 = int(max(0, self.cy - R)), int(min(h, self.cy + R + 1))
        win = f[y0:y1, x0:x1].astype(np.float32)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - self.cx, yy - self.cy)
        core = win[d <= rc]
        ring = win[(d >= self.r * 1.8) & (d <= self.r * 3.2)]
        if core.size == 0 or ring.size == 0:
            return 0.0
        return (float(core.mean()) - float(np.median(ring))) * self.pol

    def update(self, f, t):
        """returns True once, when impact is confirmed (impact_time is then set)."""
        c = self._contrast(f)
        if self.c0 is None:
            self.c0 = max(c, 12.0)
            return False
        absent = c < 0.45 * self.c0
        if not absent:
            self.run_start, self.run_n = None, 0
            self.c0 = 0.97 * self.c0 + 0.03 * max(c, 12.0)   # follow lighting
            return False
        if self.run_start is None:
            self.run_start = t
        self.run_n += 1
        if self.run_n >= 2 and t - self.run_start >= 0.025:
            self.impact_time = self.run_start
            return True
        return False

    def ball_present(self, f):
        return self.c0 is not None and self._contrast(f) >= 0.6 * self.c0


class BallTracker:
    """Phase 1 (acquire): collect candidate blobs above the tee for a short window and pick the
    chain that leaves the tee in a straight line, getting farther every frame (club = arc, body = edges).
    Phase 2 (track): predict & search a small window, adaptive frame gap for slow far-away ball."""
    MAX_GAP_SEC = 0.12
    MAX_BUF = 61
    MAX_MISS_SEC = 0.2
    ACQUIRE_SEC = 0.35

    def __init__(self, start_xy, ball_r, w, h, polarity=1, diff_thr=14):
        self.start = np.array(start_xy, float)
        self.ball_r = ball_r
        self.w, self.h = w, h
        self.polarity = polarity
        self.diff_thr = diff_thr
        self.frames, self.times = [], []
        self.pts = []             # (t, x, y)
        self.cands = []           # (t, x, y, angle, dist)  during acquisition
        self.acquired = False
        self.t_first = None
        self.miss = 0
        self.done = False
        self.model = None         # fitted flight curve used as a gate once enough points exist

    # ---------- motion model
    def velocity(self):
        q = np.array(self.pts[-4:])
        if len(q) < 2 or q[-1, 0] - q[0, 0] < 1e-6:
            return np.zeros(2)
        tt = q[:, 0] - q[-1, 0]
        return np.array([np.polyfit(tt, q[:, 1], 1)[0], np.polyfit(tt, q[:, 2], 1)[0]])

    def speed(self):
        return float(np.hypot(*self.velocity()))

    def predict(self, t):
        last = np.array(self.pts[-1][1:])
        return last + self.velocity() * (t - self.pts[-1][0])

    def max_gap(self):
        if len(self.times) < 2:
            return 1
        fdt = max(self.times[-1] - self.times[-2], 1e-4)
        return int(max(1, min((self.MAX_BUF - 1) // 2, round(self.MAX_GAP_SEC / fdt))))

    def gap(self):
        if not self.acquired or len(self.pts) < 2 or len(self.times) < 2:
            return 1
        fdt = self.times[-1] - self.times[-2]
        v = self.speed() * fdt
        need = 4 * self.ball_r + 6   # blur streaks of neighbouring frames overlap: need a clear gap
        return int(min(self.max_gap(), max(1, np.ceil(need / max(v, 0.05)))))

    # ---------- blob helpers
    def _blobs(self, a, b, c, x0, x1, y0, y1):
        """3-frame difference blobs in a window: list of (cx, cy, area, mean_db, xs, ys) in frame coords"""
        db = (b - a)[y0:y1, x0:x1]
        mask = ((np.abs(db) > self.diff_thr) &
                (np.abs((c - b)[y0:y1, x0:x1]) > self.diff_thr)).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            ys, xs = np.nonzero(labels == i)
            out.append((cents[i][0] + x0, cents[i][1] + y0, stats[i, cv2.CC_STAT_AREA],
                        float(db[ys, xs].mean()), xs + x0, ys + y0))
        return out

    def _moments(self, xs, ys, cx, cy):
        if len(xs) < 4:
            return np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]])
        return np.linalg.eigh(np.cov(np.vstack([xs - cx, ys - cy])))

    def _size_ok(self, area, ev):
        if area <= max(30, (self.ball_r * 4) ** 2):
            return True
        thick = 4 * np.sqrt(max(ev[0], 0))
        return thick <= 2.5 * self.ball_r + 3 and ev[1] >= 9 * max(ev[0], 0.25)   # thin blur streak

    def _lead(self, xs, ys, cx, cy, ev, evec, motion):
        """blurred ball = streak; the ball is NOW at the streak's leading end"""
        if len(xs) < 4 or ev[1] < 9 * max(ev[0], 0.25):
            return cx, cy
        major = evec[:, 1]
        if np.dot(major, motion) < 0:
            major = -major
        proj = major[0] * (xs - cx) + major[1] * (ys - cy)
        lead = max(0.0, float(proj.max()) - self.ball_r)
        return cx + major[0] * lead, cy + major[1] * lead

    def _club_zone(self, x, y):
        """within the club's reach the darker moving thing is the club, not the ball"""
        return np.hypot(x - self.start[0], y - self.start[1]) < 0.35 * self.h

    # ---------- main
    def push(self, f, t):
        if self.done:
            return
        if self.t_first is None:
            self.t_first = t
        self.frames.append(f.astype(np.int16)); self.times.append(t)
        if len(self.frames) > self.MAX_BUF:
            self.frames.pop(0); self.times.pop(0)
        if not self.acquired:
            if len(self.frames) >= 3:
                self._acquire_step()
            if not self.acquired and t - self.t_first > self.ACQUIRE_SEC:
                self.done = True
            return
        g = self.gap()
        if len(self.frames) < 2 * g + 1:
            return
        a, b, c = self.frames[-1 - 2 * g], self.frames[-1 - g], self.frames[-1]
        tb = self.times[-1 - g]
        if tb <= self.pts[-1][0]:
            return
        last = np.array(self.pts[-1][1:])
        full = self.predict(tb)
        pred = last + 0.75 * (full - last)
        step = np.hypot(*(full - last))
        base = max(8.0, step * 0.45 + 6)
        rad = min(base + 1.0 * self.miss, 2 * base)
        x0, x1 = int(max(0, pred[0] - rad)), int(min(self.w, pred[0] + rad + 1))
        y0, y1 = int(max(0, pred[1] - rad)), int(min(self.h, pred[1] + rad + 1))
        if x1 <= x0 or y1 <= y0:
            return self._miss()
        motion = self.velocity()
        best, bestd = None, 1e9
        for cx, cy, area, mdb, xs, ys in self._blobs(a, b, c, x0, x1, y0, y1):
            ev, evec = self._moments(xs, ys, cx, cy)
            if not self._size_ok(area, ev):
                continue
            if self._club_zone(cx, cy) and mdb * self.polarity <= 0:
                continue  # darker thing near the tee = club / body
            d = np.hypot(cx - pred[0], cy - pred[1])
            if d > rad or d >= bestd:
                continue
            best, bestd = self._lead(xs, ys, cx, cy, ev, evec, motion), d
        if best is not None and self.model is not None:
            f_, t0_ = self.model
            nu, nv, dd = f_(tb - t0_)
            if dd <= 0.05 or np.hypot(nu / dd - best[0], nv / dd - best[1]) > max(10.0, rad) * (1 + 0.5 * self.miss):
                best = None   # off the flight curve: noise / something else
        if best is None:
            self._miss()
        else:
            self.pts.append((tb, best[0], best[1])); self.miss = 0
            # build the gate only once the flight is long enough for the curve to be trustworthy
            if len(self.pts) >= 10 and len(self.pts) % 5 == 0 and self.pts[-1][0] - self.pts[0][0] >= 0.3:
                good = _robust(tuple(self.start), self.pts)
                self.model = _fit_model(tuple(self.start), good) if len(good) >= 8 else None

    def _acquire_step(self):
        a, b, c = self.frames[-3], self.frames[-2], self.frames[-1]
        tb = self.times[-2]
        sx, sy = self.start
        x0, x1 = int(max(0, sx - 0.35 * self.w)), int(min(self.w, sx + 0.35 * self.w + 1))
        y0, y1 = int(max(0, sy - 0.7 * self.h)), int(min(self.h, sy - self.ball_r * 1.5))
        if x1 <= x0 or y1 <= y0:
            return
        for cx, cy, area, mdb, xs, ys in self._blobs(a, b, c, x0, x1, y0, y1):
            if mdb * self.polarity <= 0:
                continue
            ev, evec = self._moments(xs, ys, cx, cy)
            if not self._size_ok(area, ev):
                continue
            radial = np.array([cx - sx, cy - sy])
            nr = np.hypot(*radial)
            if nr < 1:
                continue
            if ev[1] >= 4 * max(ev[0], 0.25) and abs(np.dot(evec[:, 1], radial) / nr) < np.cos(np.radians(25)):
                continue  # elongated but not pointing at the tee (shaft, body edge)
            lx, ly = self._lead(xs, ys, cx, cy, ev, evec, radial)
            self.cands.append((tb, lx, ly, np.degrees(np.arctan2(ly - sy, lx - sx)), np.hypot(lx - sx, ly - sy)))
        self._try_chain()

    def _try_chain(self):
        cs = self.cands
        if len(cs) < 3:
            return
        fdt = self.times[-1] - self.times[-2] if len(self.times) >= 2 else 0.03
        need = 4 if fdt < 1 / 50 else 3
        best = None
        for seed in cs:
            group = {}
            for cnd in cs:
                da = abs(cnd[3] - seed[3])
                if da < 4.0:
                    k = round(cnd[0], 6)
                    if k not in group or abs(group[k][3] - seed[3]) > da:
                        group[k] = cnd
            if len(group) < need:
                continue
            seq = sorted(group.values())
            # longest chain with strictly increasing distance from the tee
            L = [1] * len(seq); P = [-1] * len(seq)
            for i in range(len(seq)):
                for j in range(i):
                    if seq[j][4] + 0.5 < seq[i][4] and L[j] + 1 > L[i]:
                        L[i], P[i] = L[j] + 1, j
            i = int(np.argmax(L))
            if L[i] < need:
                continue
            chain = []
            while i >= 0:
                chain.append(seq[i]); i = P[i]
            chain.reverse()
            spread = float(np.std([q[3] for q in chain]))
            score = len(chain) - 0.2 * spread
            if best is None or score > best[0]:
                best = (score, chain)
        if best is not None:
            self.pts = [(q[0], q[1], q[2]) for q in best[1]]
            self.acquired = True
            self.cands = []

    def _miss(self):
        self.miss += 1
        if self.times[-1] - self.pts[-1][0] > self.MAX_MISS_SEC + self.MAX_GAP_SEC:
            self.done = True


def _phi(t, k):
    return t if k < 1e-6 else (1.0 - np.exp(-k * t)) / k


def _fit_model(start_xy, pts):
    """Projective ballistic model with drag:
         u(t) = (A0 + A1*phi + A2*t^2) / (1 + a*phi),   v(t) = (B0 + B1*phi + B2*t^2) / (1 + a*phi)
         phi(t) = (1 - e^{-kt}) / k     (k = drag, grid searched; rest is linear least squares)
    """
    sx, sy = start_xy
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

    return f, t0


def _sample(f, sy, w, h, landing_fraction=0.3, t_max=8.0, dt=1 / 120):
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


def _fit_once(start_xy, pts, w, h, **kw):
    if len(pts) < 3:
        return [tuple(start_xy)] + [(p[1], p[2]) for p in pts]
    f, _ = _fit_model(start_xy, pts)
    return _sample(f, start_xy[1], w, h, **kw)


def _robust(start_xy, pts):
    """global outlier rejection: refit without the worst point while it is clearly off the curve"""
    pts = list(pts)
    for _ in range(max(0, len(pts) // 3)):
        if len(pts) < 6:
            break
        f, t0 = _fit_model(start_xy, pts)
        res = []
        for (t, x, y) in pts:
            nu, nv, d = f(t - t0)
            res.append(np.hypot(nu / d - x, nv / d - y) if d > 0.05 else 1e9)
        res = np.array(res)
        k = int(np.argmax(res))
        if res[k] <= max(6.0, 4.0 * float(np.median(res))):
            break
        pts.pop(k)
    return pts


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
    if len(pts) >= 6:
        pts = _robust(start_xy, pts)
    return _fit_once(start_xy, pts, w, h, **kw)
