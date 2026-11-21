#!/usr/bin/env python3
"""AstraLink / Theseus - SIH 2026 (SIH26169) virtual-camera FSOC coarse-alignment simulation.

Examples
  python main.py                                   # 60 s mixed-motion run, haze, with secure link
  python main.py --scenario aggressive --weather fog --video out/run.mp4
  python main.py --runs 20 --scenario mixed        # Monte-Carlo over seeds, aggregated pass-rate
  python main.py --show                            # live window (ESC to stop)
"""
import argparse
import csv
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astralink.config import Config, WEATHER
from astralink.runner import run


def dashboard(L, s, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = L["t"]
    fig, ax = plt.subplots(3, 2, figsize=(14, 10))
    ax = ax.ravel()
    code = {"LOCKED": 2, "COASTING": 1, "LOST": 0}
    ax[0].step(t, [code[x] for x in L["status"]], where="post", color="tab:green")
    ax[0].set_yticks([0, 1, 2]); ax[0].set_yticklabels(["LOST", "COASTING", "LOCKED"])
    ax[0].fill_between(t, 0, 2.2, where=L["occluded"].astype(bool), color="gray", alpha=0.3, label="beacon occluded")
    ax[0].set_title("Tracker status"); ax[0].legend(loc="lower right")
    ax[1].plot(t, L["err_true"], lw=0.8, label="true beacon offset from boresight")
    ax[1].plot(t, L["err_est"], lw=0.8, label="estimator error")
    ax[1].axhline(cfg.targets.tracking_err_px, color="r", ls="--", label="target 10 px")
    ax[1].set_ylim(0, 60); ax[1].set_title("Image-plane error (px)"); ax[1].legend()
    ax[2].plot(t, np.degrees(L["omega_true"]) * 0 + [1 if m == "CT" else 0 for m in L["model"]], drawstyle="steps-post", label="selected model (1=CT, 0=CV)")
    ax[2].plot(t, (np.abs(L["omega_true"]) > 0.05).astype(float) * 0.5, drawstyle="steps-post", alpha=0.6, label="truth: target turning")
    ax[2].set_title("Adaptive CV/CT selection vs. true maneuver"); ax[2].legend()
    ax[3].plot(t, L["az_g"], label="gimbal az"); ax[3].plot(t, L["el_g"], label="gimbal el")
    ax[3].set_title("Gimbal pose (deg)"); ax[3].legend()
    ax[4].plot(t, L["conf"], lw=0.7); ax[4].set_title("Detection confidence of the accepted beacon")
    ax[5].plot(t, L["p_loss"], color="tab:red", lw=0.8)
    ax[5].set_title("Optical-link frame-loss probability (driven by real pointing error)")
    for a in ax: a.set_xlabel("time (s)"); a.grid(alpha=0.3)
    verdict = "  ".join(f"{k}:{'PASS' if v else 'FAIL'}" for k, v in s["pass"].items())
    fig.suptitle(f"{cfg.scenario.kind}/{cfg.scenario.weather}   {verdict}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fmt(s, T):
    lines = [
        f"  Acquisition time     : {s['acquisition_s']:.2f} s        (target <= {T.acquisition_s} s)   {'PASS' if s['pass']['acquisition'] else 'FAIL'}",
        f"  Tracking error (RMS) : {s['tracking_rms_px']:.2f} px      (target <= {T.tracking_err_px} px; p95 {s['tracking_p95_px']:.1f})   {'PASS' if s['pass']['tracking'] else 'FAIL'}",
        f"  Lock-loss rate       : {100*s['lock_loss_rate']:.2f} %       (target < {100*T.lock_loss_rate:.0f} %; non-LOCKED {100*s['not_locked_rate']:.1f} %)   {'PASS' if s['pass']['lock_loss'] else 'FAIL'}",
        f"  Reacquisition (p95)  : {s['reacq_p95_s']:.2f} s        (target <= {T.reacq_s} s; mean {s['reacq_mean_s']:.2f}, events {s['reacq_events']})   {'PASS' if s['pass']['reacquisition'] else 'FAIL'}",
        f"  Processing speed     : {s['fps_mean']:.0f} FPS     (target >= {T.fps:.0f} FPS; p95-latency rate {s['fps_p95_latency']:.0f})   {'PASS' if s['pass']['fps'] else 'FAIL'}",
    ]
    if "link" in s:
        k = s["link"]
        lines.append(f"  Secure link          : {k['tls_version']}, handshake {k['handshake_done_s']}, delivered "
                     f"{k['app_bytes_delivered']}/{k['app_bytes_sent']} B, integrity {'OK' if k['integrity_ok'] else 'FAILED'}, "
                     f"goodput {k['goodput_kBps']:.0f} kB/s, retx {100*k['retx_ratio']:.1f} %")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mixed", choices=["straight", "circular", "mixed", "aggressive"])
    ap.add_argument("--weather", default="haze", choices=list(WEATHER))
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1, help="Monte-Carlo runs (seed, seed+1, ...)")
    ap.add_argument("--cue-sigma", type=float, default=6.0, help="std-dev (deg) of the coarse a-priori pointing cue error")
    ap.add_argument("--out", default="results")
    ap.add_argument("--video", action="store_true", help="save an annotated mp4 of the virtual camera")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--no-comms", action="store_true", help="skip the TLS 1.3 + Selective-Repeat link")
    a = ap.parse_args()
    cfg = Config()
    cfg.scenario.kind, cfg.scenario.weather, cfg.scenario.duration_s = a.scenario, a.weather, a.duration
    cfg.scenario.cue_sigma_deg = a.cue_sigma
    os.makedirs(a.out, exist_ok=True)
    allres = []
    for i in range(a.runs):
        seed = a.seed + i
        print(f"[run {i+1}/{a.runs}] scenario={a.scenario} weather={a.weather} seed={seed}", flush=True)
        vp = os.path.join(a.out, f"run_seed{seed}.mp4") if a.video else None
        L, s = run(cfg, seed, video_path=vp, show=a.show, comms=not a.no_comms, progress=(a.runs == 1))
        print(fmt(s, cfg.targets))
        allres.append(s)
        tag = f"{a.scenario}_{a.weather}_seed{seed}"
        dashboard(L, s, os.path.join(a.out, f"dashboard_{tag}.png"), cfg)
        with open(os.path.join(a.out, f"log_{tag}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(L.keys())
            for row in zip(*[L[k] for k in L]):
                w.writerow(row)
        json.dump(s, open(os.path.join(a.out, f"summary_{tag}.json"), "w"), indent=2, default=float)
    if a.runs > 1:
        print("\n=== Monte-Carlo aggregate over", a.runs, "runs ===")
        keys = list(allres[0]["pass"])
        per_metric = {}
        for k in keys:
            pr = 100 * float(np.mean([r["pass"][k] for r in allres]))
            per_metric[k] = round(pr, 1)
            print(f"  {k:14s} pass-rate {pr:5.1f} %")
        allpass = [all(r["pass"][k] for k in keys) for r in allres]
        ap_rate = 100 * float(np.mean(allpass))
        fail_seeds = [a.seed + i for i, ok in enumerate(allpass) if not ok]
        print(f"  {'ALL metrics':14s} pass-rate {ap_rate:5.1f} %   (failing seeds: {fail_seeds})")
        medians, worsts = {}, {}
        for k in ("acquisition_s", "tracking_rms_px", "lock_loss_rate", "reacq_p95_s", "fps_mean"):
            v = np.array([r[k] for r in allres], float)
            if k == "fps_mean":
                med, worst = float(np.nanmedian(v)), float(np.nanmin(v))
            else:
                med, worst = float(np.nanmedian(v)), float(np.nanmax(v))
            medians[k], worsts[k] = round(med, 3), round(worst, 3)
            print(f"  {k:16s} median {med:8.3f}  worst {worst:8.3f}")
        summary = dict(scenario=a.scenario, weather=a.weather, n_runs=a.runs, seed0=a.seed,
                       per_metric_pass_rate=per_metric, all_metrics_pass_rate=round(ap_rate, 1),
                       failing_seeds=fail_seeds, medians=medians, worsts=worsts)
        json.dump(summary, open(os.path.join(a.out, f"summary_{a.scenario}_{a.weather}.json"), "w"), indent=2, default=float)
        json.dump(allres, open(os.path.join(a.out, f"montecarlo_{a.scenario}_{a.weather}.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
