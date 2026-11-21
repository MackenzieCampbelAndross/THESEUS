"""Simulation harness: runs scene + pipeline (+ optical link), logs everything, scores against targets."""
import math
import time
import numpy as np
import cv2
from .config import Config
from .world import VirtualScene, Gimbal
from .alignment import CoarseAligner, LOCKED, COASTING, LOST


def run(cfg: Config, seed=0, video_path=None, show=False, comms=True, progress=False):
    scene = VirtualScene(cfg, seed)
    rng = np.random.default_rng(seed + 10_000)
    dt = 1.0 / cfg.camera.fps
    n = int(cfg.scenario.duration_s * cfg.camera.fps)
    gimbal = Gimbal(cfg, scene.cue[0], scene.cue[1], rng)     # starts at the coarse cue only
    aligner = CoarseAligner(cfg, rng, scene.px_per_deg, scene.cx, scene.cy, scene.f, scene.cue)
    aligner.det.calibrate(scene.dark_frame())
    link = None
    if comms and cfg.comms.enabled:
        from .comms import OpticalLink
        link = OpticalLink(cfg.comms, seed)
    vw = None
    if video_path:
        vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.camera.fps, (cfg.camera.width, cfg.camera.height))
    L = {k: [] for k in ["t", "status", "err_true", "err_est", "visible", "conf", "model", "omega_true",
                         "proc_ms", "az_g", "el_g", "p_loss", "occluded", "ndet"]}
    for k in range(n):
        t = k * dt
        true_pose = gimbal.ang.copy()
        frame, tr = scene.capture(k, true_pose)
        enc = gimbal.read()
        t0 = time.perf_counter()
        v, out = aligner.process(frame, enc)
        proc = (time.perf_counter() - t0) * 1000
        gimbal.command(*v)
        gimbal.step(dt)
        # ---- scoring (ground truth used ONLY here) ----
        err_true = math.hypot(tr["u"] - scene.cx, tr["v"] - scene.cy) if tr["visible"] else float("nan")
        du = scene.cx + tr["f"] * math.tan(math.radians((out.est_az - enc[0]))) if out.tracked else float("nan")
        dv = scene.cy - tr["f"] * math.tan(math.radians((out.est_el - enc[1]))) if out.tracked else float("nan")
        err_est = math.hypot(du - tr["u"], dv - tr["v"]) if (out.tracked and tr["visible"]) else float("nan")
        coupling = math.exp(-(err_true / cfg.comms.coupling_sigma_px) ** 2) if tr["visible"] else 0.0
        p_loss = 1 - (1 - cfg.comms.base_loss) * coupling
        if link is not None:
            link.advance_frame(dt, p_loss)
        for key, val in zip(L.keys(), [t, out.status, err_true, err_est, tr["visible"], out.conf, out.model,
                                       tr["omega"], proc, true_pose[0], true_pose[1], p_loss, tr["occluded"], out.ndet]):
            L[key].append(val)
        if vw is not None or show:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            col = {LOCKED: (0, 200, 0), COASTING: (0, 200, 255), LOST: (0, 0, 255)}[out.status]
            cv2.drawMarker(vis, (int(scene.cx), int(scene.cy)), (255, 255, 0), cv2.MARKER_CROSS, 24, 1)
            if out.tracked and not math.isnan(du):
                cv2.circle(vis, (int(du), int(dv)), 14, col, 2)
            cv2.putText(vis, f"{out.status} {out.model} t={t:5.1f}s conf={out.conf:.2f}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            if vw is not None:
                vw.write(vis)
            if show:
                cv2.imshow("AstraLink virtual camera", vis)
                if cv2.waitKey(1) == 27:
                    break
        if progress and k % 300 == 0:
            print(f"  t={t:5.1f}s  {out.status:8s} {out.model}", flush=True)
    if vw is not None:
        vw.release()
    if show:
        cv2.destroyAllWindows()
    for key in L:
        L[key] = np.array(L[key])
    return L, score(L, cfg, dt, link)


def score(L, cfg, dt, link):
    T = cfg.targets
    st = L["status"]
    locked = st == LOCKED
    first = np.argmax(locked) if locked.any() else -1
    # consecutive confirmed lock (>= 5 frames) counts as acquired
    acq_idx = -1
    good = locked & (np.nan_to_num(L["err_est"], nan=1e9) <= 15.0)   # truth check: lock must be on the REAL beacon
    for i in range(len(st) - 5):
        if good[i:i + 5].all():
            acq_idx = i
            break
    acq_t = L["t"][acq_idx] if acq_idx >= 0 else float("inf")
    post = np.arange(len(st)) >= max(acq_idx, 0)
    settle = np.arange(len(st)) >= (max(acq_idx, 0) + int(round(1.0 / dt)))   # skip 1 s pull-in for steady-state error
    if acq_idx < 0:
        post[:] = True
    vis = L["visible"].astype(bool)
    lost_true = (~vis) | (st == LOST)
    lock_loss = float(lost_true[post].mean()) if post.any() else 1.0
    not_locked = float((~locked[post]).mean()) if post.any() else 1.0
    e = L["err_true"][settle & vis & locked]
    trk_rms = float(np.sqrt(np.nanmean(e ** 2))) if e.size else float("nan")
    trk_p95 = float(np.nanpercentile(e, 95)) if e.size else float("nan")
    ee = L["err_est"][post & locked]
    est_rms = float(np.sqrt(np.nanmean(ee ** 2))) if np.isfinite(ee).any() else float("nan")
    # reacquisition episodes: first non-LOCKED frame after LOCKED -> next LOCKED frame
    episodes, start = [], None
    for i in range(acq_idx if acq_idx >= 0 else 0, len(st)):
        if start is None and st[i] != LOCKED and acq_idx >= 0 and i > acq_idx:
            start = i
        elif start is not None and st[i] == LOCKED:
            avail = [j for j in range(start, i) if vis[j] and not L["occluded"][j]]
            eff = avail[0] if avail else i          # count from when the beacon was physically observable again
            episodes.append((i - eff) * dt)
            start = None
    unresolved = 1 if start is not None else 0
    long_eps = [x for x in episodes if x > 3 * dt]      # ignore single-frame flickers
    reacq_mean = float(np.mean(long_eps)) if long_eps else 0.0
    reacq_p95 = float(np.percentile(long_eps, 95)) if long_eps else 0.0
    proc = L["proc_ms"]
    fps = 1000.0 / float(np.mean(proc[5:])) if len(proc) > 10 else float("nan")
    fps_p95 = 1000.0 / float(np.percentile(proc[5:], 95)) if len(proc) > 10 else float("nan")
    s = dict(acquisition_s=float(acq_t), tracking_rms_px=trk_rms, tracking_p95_px=trk_p95, estimator_rms_px=est_rms,
             lock_loss_rate=lock_loss, not_locked_rate=not_locked, reacq_events=len(long_eps) + unresolved,
             reacq_mean_s=reacq_mean, reacq_p95_s=reacq_p95, reacq_unresolved=unresolved,
             fps_mean=fps, fps_p95_latency=fps_p95,
             ct_fraction=float(np.mean(L["model"][post] == "CT")) if post.any() else 0.0)
    s["pass"] = dict(acquisition=bool(acq_t <= T.acquisition_s), tracking=bool(trk_rms <= T.tracking_err_px),
                     lock_loss=bool(lock_loss < T.lock_loss_rate),
                     reacquisition=bool(reacq_p95 <= T.reacq_s and not unresolved), fps=bool(fps >= T.fps))
    if link is not None:
        s["link"] = link.report(cfg.scenario.duration_s)
    return s
