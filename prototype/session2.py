"""Python mirror of ShotSession v2 (same logic goes into ShotSession.kt)."""
from collections import deque
from engine2 import refine_ball, ImpactDetector, BallTracker, fit_trajectory

GIVE_UP_SEC = 0.6


class Session:
    def __init__(self, first_frame, tap):
        self.h, self.w = first_frame.shape
        self.x, self.y, self.r, self.pol = refine_ball(first_frame, tap, self.w)
        self.det = ImpactDetector((self.x, self.y), self.r, self.pol)
        self.hist = deque(maxlen=64)
        self.state = 'ARMED'
        self.trk = None
        self.impact = None
        self.rearms = 0

    def feed(self, f, t):
        self.hist.append((f, t))
        if self.state == 'ARMED':
            if self.det.update(f, t):
                self.impact = self.det.impact_time
                self.trk = BallTracker((self.x, self.y), self.r, self.w, self.h, self.pol)
                # replay buffered frames from just before the ball vanished
                dt = (t - self.hist[0][1]) / max(1, len(self.hist) - 1) if len(self.hist) > 1 else 0.03
                for hf, ht in self.hist:
                    if ht >= self.impact - 1.5 * dt:
                        self.trk.push(hf, ht)
                self.state = 'TRACKING'
        elif self.state == 'TRACKING':
            self.trk.push(f, t)
            if not self.trk.acquired and (self.trk.done or t - self.impact > GIVE_UP_SEC):
                if self.det.ball_present(f):        # false alarm: ball is still on the tee
                    self.det = ImpactDetector((self.x, self.y), self.r, self.pol)
                    self.det.update(f, t)
                    self.state, self.trk, self.rearms = 'ARMED', None, self.rearms + 1
                else:
                    self.state = 'FAILED'
            elif self.trk.done:
                self.finish()

    def finish(self):
        if self.state != 'TRACKING':
            return
        pts = self.trk.pts
        self.curve = fit_trajectory((self.x, self.y), pts, self.w, self.h) if len(pts) >= 3 else None
        self.state = 'FINISHED' if self.curve else 'FAILED'


def run(sc):
    s = None
    for f, t in zip(sc['frames'], sc['times']):
        if s is None:
            s = Session(f, sc['tap'])
        s.feed(f, t)
        if s.state in ('FINISHED', 'FAILED'):
            break
    if s.state == 'TRACKING':
        s.finish()
    pts = s.trk.pts if s.trk else []
    return s.impact, pts, (s.curve if s.state == 'FINISHED' else None)
