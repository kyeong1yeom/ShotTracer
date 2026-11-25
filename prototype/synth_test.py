"""Generate a synthetic behind-the-golfer shot and run the engine on it."""
import numpy as np, cv2, sys
from engine import ImpactDetector, BallTracker, fit_trajectory

W, H, FPS = 360, 640, int(sys.argv[1]) if len(sys.argv) > 1 else 120
N = int(FPS * 3.5)
VIS = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '-v' else 99
IMPACT = int(FPS * 0.5)
rng = np.random.default_rng(1)
f_px = 520.0
cam = np.array([0.0, 1.5, -3.0])  # x, y(up), z
horizon = H * 0.48

def project(p):
    r = p - cam
    return np.array([W / 2 + f_px * r[0] / r[2], horizon - f_px * r[1] / r[2]]), f_px / r[2]

# ball 3D flight with simple drag
pos, vel = np.array([0.0, 0.02, 0.0]), None
v0, la, side = 62.0, np.radians(15), np.radians(3)
vel = v0 * np.array([np.sin(side) * np.cos(la), np.sin(la), np.cos(side) * np.cos(la)])
truth = []
traj3d = [pos.copy()]
dt = 1.0 / FPS
for i in range(N):
    sp = np.linalg.norm(vel)
    acc = np.array([0, -9.81, 0]) - 0.0075 * sp * vel + np.array([0, 0.0028 * sp * sp * 0.6, 0])
    vel = vel + acc * dt
    pos = pos + vel * dt
    if pos[1] < 0: break
    traj3d.append(pos.copy())

base = np.zeros((H, W), np.float32)
for yy in range(H):
    base[yy] = 195 - 40 * yy / horizon if yy < horizon else 95 - 10 * (yy - horizon) / (H - horizon)
start2d, _ = project(traj3d[0])

def frame(i):
    f = base.copy() + rng.normal(0, 3, (H, W))
    # golfer body + swinging club (left of ball)
    gx = int(start2d[0] - 45)
    cv2.rectangle(f, (gx - 25, int(start2d[1] - 260)), (gx + 25, int(start2d[1] + 20)), 40, -1)
    ang = np.pi * (0.5 + 1.6 * (i - IMPACT) / (FPS * 0.6))
    ang = np.clip(ang, -1.2, 4.2)
    hand = np.array([gx + 10, start2d[1] - 120])
    head = hand + 125 * np.array([np.cos(ang), np.sin(ang)])
    cv2.line(f, tuple(hand.astype(int)), tuple(head.astype(int)), 30, 3)
    k = i - IMPACT
    if k < 0: p, s = project(traj3d[0])
    elif k < len(traj3d): p, s = project(traj3d[k])
    else: return f.clip(0, 255).astype(np.uint8), None
    if k > VIS * FPS: return f.clip(0, 255).astype(np.uint8), None
    rad = max(1.0, s * 0.021 * 1.6)
    cv2.circle(f, (int(round(p[0] * 4)), int(round(p[1] * 4))), int(rad * 4), 245, -1, cv2.LINE_AA, shift=2)
    return f.clip(0, 255).astype(np.uint8), (p if k >= 0 else None)

frames, gts = zip(*[frame(i) for i in range(N)])
ball_r = max(1.5, project(traj3d[0])[1] * 0.021 * 1.6)
det = ImpactDetector(start2d, ball_r)
trk = None
impact_frame = None
for i, f in enumerate(frames):
    t = i / FPS
    if trk is None:
        if det.update(f):
            impact_frame = i
            trk = BallTracker(start2d, ball_r, W, H, det.polarity)
            trk.push(frames[i - 1], t - 1 / FPS)
            trk.push(f, t)
    else:
        trk.push(f, t)
        if trk.done: break

print("impact detected at", impact_frame, "true", IMPACT)
errs = []
for (t, x, y) in trk.pts:
    g = gts[int(round(t * FPS))]
    if g is not None: errs.append(np.hypot(x - g[0], y - g[1]))
print(f"tracked {len(trk.pts)} pts, median err {np.median(errs):.2f}px, max {np.max(errs):.2f}px")
curve = fit_trajectory(start2d, trk.pts, W, H)
truth2d = [project(p)[0] for p in traj3d]
apex_true = min(p[1] for p in truth2d)
apex_fit = min(p[1] for p in curve)
print(f"apex y true {apex_true:.1f} fit {apex_fit:.1f}; tracked span {trk.pts[-1][0]-trk.pts[0][0]:.2f}s")

out = cv2.cvtColor(frames[IMPACT + 5], cv2.COLOR_GRAY2BGR)
cv2.polylines(out, [np.array(truth2d, np.int32)], False, (0, 255, 0), 1)
cv2.polylines(out, [np.array(curve, np.int32)], False, (0, 0, 255), 2)
for _, x, y in trk.pts: cv2.circle(out, (int(x), int(y)), 2, (255, 0, 0), -1)
cv2.imwrite(f"result_{FPS}_{VIS}.png", out)
land_t=truth2d[-1]; print(f'landing true ({land_t[0]:.0f},{land_t[1]:.0f}) fit ({curve[-1][0]:.0f},{curve[-1][1]:.0f})')
if '-v' in sys.argv:
    for (t,x,y) in [p for p in trk.pts if p[0]*FPS>IMPACT+VIS*FPS-3][:12]:
        g=gts[int(round(t*FPS))]; print(f"{t*FPS:.0f} det({x:.0f},{y:.0f})", "" if g is None else f"gt({g[0]:.0f},{g[1]:.0f})")
