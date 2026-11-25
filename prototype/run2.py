"""Run a ShotSession-equivalent driver on synth2 scenes and report."""
import sys, numpy as np, cv2
import synth2


def run_old(sc):
    from engine import ImpactDetector, BallTracker, fit_trajectory
    W = synth2.W
    ball_r = max(2.0, W * 0.01)
    det = ImpactDetector(sc['tap'], ball_r); trk = None; prev = None; imp = None
    for f, t in zip(sc['frames'], sc['times']):
        if trk is None:
            if det.update(f):
                imp = t
                trk = BallTracker(sc['tap'], ball_r, W, synth2.H, det.polarity)
                if prev is not None: trk.push(prev, t - 1 / sc['fps'])
                trk.push(f, t)
            prev = f
        else:
            trk.push(f, t)
            if trk.done: break
    pts = trk.pts if trk else []
    curve = fit_trajectory(sc['tap'], pts, W, synth2.H) if len(pts) >= 3 else None
    return imp, pts, curve


def report(name, sc, imp, pts, curve, img=None):
    good = 0; errs = []
    for (t, x, y) in pts:
        k = int(round((t - sc['times'][0]) * sc['fps']))
        g = sc['gt'][k] if 0 <= k < len(sc['gt']) else None
        if g is not None:
            e = np.hypot(x - g[0], y - g[1]); errs.append(e); good += e < 4
    apex_t = min(p[1] for p in sc['truth'])
    apex = f"{min(p[1] for p in curve):.0f}/{apex_t:.0f}" if curve else "-"
    land = f"({curve[-1][0]:.0f},{curve[-1][1]:.0f})/({sc['truth'][-1][0]:.0f},{sc['truth'][-1][1]:.0f})" if curve else "-"
    ok = imp is not None and abs(imp - sc['impact_t']) < 0.05 and curve is not None and abs(min(p[1] for p in curve) - apex_t) < 25
    print(f"{name:28s} impact={('%.3f' % imp) if imp else 'none':>6} (true {sc['impact_t']:.3f}) "
          f"pts={len(pts):3d} on-ball={good:3d} apex={apex} land={land}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'old'
    mod = __import__('engine2' if which == 'new' else 'engine')
    runner = run_old if which == 'old' else __import__('session2').run
    allok = True
    for fps in (240, 120, 60, 30):
        for vis in (99, 0.5):
            sc = synth2.make(fps, vis=vis)
            r = runner(sc)
            allok &= report(f"{which} {fps}fps vis={vis}", sc, *r)
    print("ALL OK" if allok else "SOME FAILED")
