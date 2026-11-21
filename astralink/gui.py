#!/usr/bin/env python3
"""AstraLink live GUI (OpenCV). Real-time virtual-camera view with detection boxes,
lock status, FPS, confidence, and a rolling tracking-error trace.

Run:
    python gui.py --scenario mixed --weather haze --seed 0     # live window (ESC to quit)
    python gui.py --sample                                      # render a few offscreen frames to results/gui_sample/ and exit
"""
import argparse
import math
import os
import time
import numpy as np
import cv2
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astralink.config import Config, WEATHER
from astralink.world import VirtualScene, Gimbal
from astralink.alignment import CoarseAligner, LOCKED, COASTING, LOST

COL = {LOCKED: (60, 220, 60), COASTING: (0, 200, 255), LOST: (0, 60, 255)}


def annotate(frame, tr, out, dets, enc, scene, err_hist, fps):
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    # every detection: small square
    for d in dets:
        cv2.rectangle(vis, (int(d.u) - 6, int(d.v) - 6), (int(d.u) + 6, int(d.v) + 6), (180, 180, 180), 1)
    # boresight crosshair
    cv2.drawMarker(vis, (int(scene.cx), int(scene.cy)), (255, 255, 0), cv2.MARKER_CROSS, 22, 1)
    col = COL[out.status]
    # tracked beacon estimate -> pixel
    if out.tracked:
        du = scene.cx + scene.f * math.tan(math.radians(out.est_az - enc[0]))
        dv = scene.cy - scene.f * math.tan(math.radians(out.est_el - enc[1]))
        if 0 <= du < w and 0 <= dv < h:
            cv2.rectangle(vis, (int(du) - 14, int(dv) - 14), (int(du) + 14, int(dv) + 14), col, 2)
            cv2.line(vis, (int(du), int(dv)), (int(scene.cx), int(scene.cy)), col, 1)
    # HUD
    err = 0.0 if not tr["visible"] else math.hypot(tr["u"] - scene.cx, tr["v"] - scene.cy)
    cv2.rectangle(vis, (0, 0), (w, 26), (30, 30, 30), -1)
    cv2.putText(vis, f"{out.status}  {out.model}  conf={out.conf:.2f}  {fps:4.0f} FPS  err={err:5.1f}px",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    # rolling error trace strip at the bottom
    strip_h = 60
    y0 = h - strip_h
    cv2.rectangle(vis, (0, y0), (w, h), (20, 20, 20), -1)
    cv2.putText(vis, "tracking error (px, 0-30)", (6, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    if len(err_hist) > 1:
        n = len(err_hist)
        pts = np.array(err_hist[-w:], float)
        xs = np.linspace(0, w - 1, len(pts)).astype(int)
        ys = (y0 + strip_h - np.clip(pts / 30.0, 0, 1) * (strip_h - 16)).astype(int)
        for i in range(1, len(pts)):
            cv2.line(vis, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), col, 1)
    return vis


def run(cfg, seed, live=True, sample_dir=None, sample_every=150, max_frames=None):
    scene = VirtualScene(cfg, seed)
    rng = np.random.default_rng(seed + 10_000)
    dt = 1.0 / cfg.camera.fps
    n = max_frames or int(cfg.scenario.duration_s * cfg.camera.fps)
    gimbal = Gimbal(cfg, scene.cue[0], scene.cue[1], rng)
    al = CoarseAligner(cfg, rng, scene.px_per_deg, scene.cx, scene.cy, scene.f, scene.cue)
    al.det.calibrate(scene.dark_frame())
    if live:
        cv2.namedWindow("AstraLink", cv2.WINDOW_NORMAL)
    if sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
    err_hist, fps = [], 1.0
    for k in range(n):
        pose = gimbal.ang.copy()
        frame, tr = scene.capture(k, pose, acq=False)
        al.f = scene.f
        enc = gimbal.read()
        t0 = time.perf_counter()
        dets = al.det.detect(frame)
        v, out = al.process(frame, enc)
        fps = 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - t0, 1e-6))
        gimbal.command(*v); gimbal.step(dt)
        err = 0.0 if not tr["visible"] else math.hypot(tr["u"] - scene.cx, tr["v"] - scene.cy)
        err_hist.append(err)
        vis = annotate(frame, tr, out, dets, enc, scene, err_hist, fps)
        if live:
            cv2.imshow("AstraLink", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if sample_dir and k % sample_every == 0:
            cv2.imwrite(os.path.join(sample_dir, f"gui_seed{seed}_k{k:05d}.png"), vis)
    if live:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mixed", choices=["straight", "circular", "mixed", "aggressive"])
    ap.add_argument("--weather", default="haze", choices=list(WEATHER))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--sample", action="store_true", help="render offscreen sample frames and exit (headless smoke test)")
    a = ap.parse_args()
    cfg = Config()
    cfg.scenario.kind, cfg.scenario.weather, cfg.scenario.duration_s = a.scenario, a.weather, a.duration
    if a.sample:
        run(cfg, a.seed, live=False, sample_dir="results/gui_sample", max_frames=600)
        print("wrote samples to results/gui_sample/")
    else:
        run(cfg, a.seed, live=True)


if __name__ == "__main__":
    main()
