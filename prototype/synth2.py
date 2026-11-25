"""More realistic synthetic down-the-line shot:
 - club head rests next to the ball at address and waggles, then backswing / downswing
 - follow-through swings the club head UP across the area above the tee (decoy)
 - ball has motion blur (streak, lower contrast) and realistic size (~3.6px radius)
 - the user's tap is a few px off the ball centre
make(fps) -> dict(frames, times, gt, tee, tap, W, H, impact_t)"""
import numpy as np, cv2

W, H = 360, 640


def make(fps=120, seed=1, vis=99.0, tap_err=(5, -4), club=True, blur=True):
    rng = np.random.default_rng(seed)
    f_px, cam, horizon = 520.0, np.array([0.0, 1.5, -3.0]), H * 0.48

    def project(p):
        r = p - cam
        return np.array([W / 2 + f_px * r[0] / r[2], horizon - f_px * r[1] / r[2]]), f_px / r[2]

    # 3D ball flight
    dt = 1.0 / fps
    pos = np.array([0.0, 0.02, 0.0])
    v0, la, side = 62.0, np.radians(15), np.radians(3)
    vel = v0 * np.array([np.sin(side) * np.cos(la), np.sin(la), np.cos(side) * np.cos(la)])
    traj = [pos.copy()]
    for _ in range(int(fps * 8)):
        sp = np.linalg.norm(vel)
        acc = np.array([0, -9.81, 0]) - 0.0075 * sp * vel + np.array([0, 0.0028 * sp * sp * 0.6, 0])
        vel = vel + acc * dt; pos = pos + vel * dt
        if pos[1] < 0: break
        traj.append(pos.copy())
    tee, s0 = project(traj[0])
    ball_r = s0 * 0.021

    base = np.zeros((H, W), np.float32)
    for yy in range(H):
        base[yy] = 195 - 40 * yy / horizon if yy < horizon else 95 - 10 * (yy - horizon) / (H - horizon)
    # static texture on the grass (real grass isn't flat)
    tex = cv2.GaussianBlur(rng.normal(0, 6, (H, W)).astype(np.float32), (0, 0), 1.2)
    base[int(horizon):] += tex[int(horizon):]

    hands = np.array([tee[0] - 60, tee[1] - 150])
    rad_c = np.hypot(*(tee - hands)) - 8   # club head rests just behind the ball
    a0 = np.arctan2(tee[1] + 4 - hands[1], tee[0] - 3 - hands[0])

    def club_angle(t):
        if t < -1.5:  # address + waggle
            return a0 + 0.035 * np.sin(2 * np.pi * 3 * t) * (t < -1.7)
        if t < -0.3:  # backswing (rotate "back", away from the target line)
            u = (t + 1.5) / 1.2
            return a0 + np.radians(200) * (0.5 - 0.5 * np.cos(np.pi * u))
        if t < 0:     # downswing, accelerating
            u = (t + 0.3) / 0.3
            return a0 + np.radians(200) * (1 - u * u)
        u = min(1.0, t / 0.35)  # follow-through: head swings UP past the tee area
        return a0 - np.radians(160) * (1 - (1 - u) ** 2)

    T0, T1 = -2.0, 3.0
    n = int((T1 - T0) * fps)
    frames, times, gt = [], [], []
    prev_p = None
    for i in range(n):
        t = T0 + i / fps
        f = base.copy()
        # golfer body (slight sway)
        bx = int(tee[0] - 95 + 3 * np.sin(t * 2))
        cv2.rectangle(f, (bx - 30, int(tee[1] - 330)), (bx + 30, int(tee[1] + 25)), 45, -1)
        k = int(round(t * fps))
        g = None
        if k <= 0:
            p = tee
        elif k < len(traj) and t <= vis:
            p, s = project(traj[k])
            g = p
        else:
            p = None
        if p is not None:
            r = max(0.7, (s0 if k <= 0 else s) * 0.021)
            if blur and prev_p is not None and k > 0:
                L = np.hypot(*(p - prev_p))
                inten = 95 + 150 * min(1.0, (2 * r + 1) / (L + 2 * r + 1)) + 20
                layer = np.zeros_like(f); m = np.zeros_like(f)
                cv2.line(m, tuple((prev_p * 4).astype(int)), tuple((p * 4).astype(int)), 1.0,
                         max(1, int(2 * r)) * 4 // 4 + 1, cv2.LINE_AA, shift=2)
                f = np.where(m > 0, np.maximum(f, inten * m + f * (1 - m)), f)
            else:
                cv2.circle(f, (int(round(p[0] * 4)), int(round(p[1] * 4))), int(r * 4), 245, -1, cv2.LINE_AA, shift=2)
            prev_p = p if k > 0 else tee
        if club:
            ang = club_angle(t)
            head = hands + rad_c * np.array([np.cos(ang), np.sin(ang)])
            cv2.line(f, tuple(hands.astype(int)), tuple(head.astype(int)), 75, 2, cv2.LINE_AA)
            cv2.ellipse(f, (int(head[0]), int(head[1])), (6, 4), np.degrees(ang), 0, 360, 40, -1, cv2.LINE_AA)
            cv2.circle(f, tuple(hands.astype(int)), 9, 50, -1)
        f = f + rng.normal(0, 3, (H, W))
        frames.append(f.clip(0, 255).astype(np.uint8)); times.append(t + 2.0); gt.append(g)
    tap = (tee[0] + tap_err[0], tee[1] + tap_err[1])
    truth = [project(p)[0] for p in traj]
    return dict(frames=frames, times=times, gt=gt, tee=tee, tap=tap, ball_r=ball_r,
                truth=truth, impact_t=2.0, fps=fps)
