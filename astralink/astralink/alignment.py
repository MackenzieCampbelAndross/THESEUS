"""Closed-loop coarse alignment: detect -> track/predict -> status (LOCKED/COASTING/LOST) -> gimbal rates.
Lost-beacon recovery uses a prediction-guided Levy-flight search."""
import math
from dataclasses import dataclass
import numpy as np
from .config import Config
from .detector import BeaconDetector
from .tracker import AdaptiveTracker

LOCKED, COASTING, LOST = "LOCKED", "COASTING", "LOST"


class LevySearch:
    def __init__(self, c, rng):
        self.c, self.rng = c, rng
        b = c.levy_beta
        self.sig_u = (math.gamma(1 + b) * math.sin(math.pi * b / 2) /
                      (math.gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))) ** (1 / b)
        self.reset()

    def reset(self):
        self.off = np.zeros(2)
        self.t_search = 0.0
        self.t_wp = 0.0
        self.new = True

    def _step(self):
        u = self.rng.normal(0, self.sig_u)
        v = self.rng.normal()
        L = self.c.levy_scale_deg * abs(u) / max(abs(v), 1e-6) ** (1 / self.c.levy_beta)   # Mantegna
        phi = self.rng.uniform(0, 2 * math.pi)
        return min(L, 2 * self.c.levy_r_max_deg) * np.array([math.cos(phi), math.sin(phi)])

    def waypoint(self, center, enc, dt):
        c = self.c
        self.t_search += dt
        self.t_wp += dt
        R = min(c.levy_r_max_deg, c.levy_r0_deg + c.levy_r_growth_deg_s * self.t_search)
        wp = center + self.off
        if self.new or np.linalg.norm(wp - enc) < 1.0 or self.t_wp > c.levy_dwell_s:
            self.off = self.off + self._step()
            n = np.linalg.norm(self.off)
            if n > R:
                self.off *= R / n              # confine to the (growing) search disc
            self.t_wp, self.new = 0.0, False
            wp = center + self.off
        return wp


@dataclass
class Output:
    status: str
    model: str
    conf: float
    est_az: float
    est_el: float
    pred_az: float
    pred_el: float
    ndet: int
    tracked: bool


class CoarseAligner:
    def __init__(self, cfg: Config, rng, px_per_deg, cx, cy, f_px, cue):
        self.cfg, self.c = cfg, cfg.control
        self.dt = 1.0 / cfg.camera.fps
        self.det = BeaconDetector(cfg.detector)
        self.trk = AdaptiveTracker(cfg.tracker, px_per_deg, self.dt)
        self.levy = LevySearch(self.c, rng)
        self.cx, self.cy, self.f = cx, cy, f_px
        self.status = LOST
        self.miss = 0
        self.pending = None
        self.pend_n = 0
        self.pend_hist = []
        self.pend_dom = True
        self.center = np.array(cue, float)      # where to search (coarse cue, later the prediction)
        self.acquired = False
        self.snr_ema = None                     # brightness signature of the tracked beacon
        self.clutter = []                       # learned world-fixed light sources [az, el, hits]

    def to_abs(self, d, enc):
        return np.array([enc[0] + math.degrees(math.atan((d.u - self.cx) / self.f)),
                         enc[1] - math.degrees(math.atan((d.v - self.cy) / self.f))])

    def _learn_clutter(self, zs, chosen):
        """While a beacon is being tracked, persistent detections that are NOT the beacon are static clutter."""
        tp = self.trk.pos()
        for z, d in zs:
            if (chosen is not None and d is chosen[1]) or np.linalg.norm(z - tp) < 0.8:
                continue
            for e in self.clutter:
                if math.hypot(e[0] - z[0], e[1] - z[1]) < 0.8:
                    e[0] += 0.3 * (z[0] - e[0]); e[1] += 0.3 * (z[1] - e[1]); e[2] += 1
                    break
            else:
                if len(self.clutter) < 40:
                    self.clutter.append([z[0], z[1], 1])

    def _is_clutter(self, z):
        return any(e[2] >= 8 and math.hypot(e[0] - z[0], e[1] - z[1]) < 0.8 for e in self.clutter)

    def _sig_ok(self, d):
        return self.snr_ema is None or d.snr >= 0.5 * self.snr_ema

    def _gate_deg(self):
        return min(self.c.gate_min_deg + 3.0 * self.trk.pos_sigma(), self.c.gate_max_deg)

    def process(self, frame, enc):
        c, trk = self.c, self.trk
        dets = self.det.detect(frame)
        enc_v = np.array(enc)
        zs = [(self.to_abs(d, enc_v), d) for d in dets]
        chosen = None
        if trk.ready:
            trk.predict()
            if self.status in (LOCKED, COASTING):
                pred = trk.pos()
                best = None
                for z, d in zs:
                    if d.conf >= c.c_track and np.linalg.norm(z - pred) <= self._gate_deg() and not self._is_clutter(z) and self._sig_ok(d):
                        if best is None or d.snr > best[1].snr:
                            best = (z, d)
                chosen = best
        if self.status in (LOCKED, COASTING) and chosen is not None:
            trk.update(chosen[0], chosen[1].conf)
            self.status, self.miss = LOCKED, 0
            self.snr_ema = chosen[1].snr if self.snr_ema is None else 0.9 * self.snr_ema + 0.1 * chosen[1].snr
            self.center = trk.pos()
            self._learn_clutter(zs, chosen)
        elif self.status in (LOCKED, COASTING):
            self.miss += 1
            trk.coast_damp()
            self.center = trk.pos()
            self.status = COASTING if self.miss <= c.coast_frames else LOST
            if self.status == LOST:
                self.levy.reset()
                self.pending, self.pend_n = None, 0
        else:  # LOST / initial search: confirm a strong, self-consistent detection before locking
            if trk.ready:
                trk.coast_damp()
                self.center = trk.pos()
            cand = [(z, d) for z, d in zs if d.conf >= c.c_acquire and not self._is_clutter(z) and self._sig_ok(d)]
            if cand:
                z, d = max(cand, key=lambda x: x[1].snr)
                others = [x[1].snr for x in zs if x[1] is not d]
                dominant = (d.snr >= c.dominance * float(np.median(others))) if others else (d.snr >= c.lone_snr)
                if self.pending is not None and np.linalg.norm(z - self.pending) < c.confirm_gate_deg:
                    self.pend_n += 1
                else:
                    self.pend_n, self.pend_hist, self.pend_dom = 1, [], True
                self.pend_dom = self.pend_dom and dominant
                self.pending = z
                self.pend_hist = (self.pend_hist + [z.copy()])[-c.confirm_frames:]
                moved = np.linalg.norm(z - self.pend_hist[0]) if len(self.pend_hist) >= c.confirm_frames else 0.0
                # a real (mobile) beacon either moves in world angle or is a clear brightness outlier;
                # a static background light does neither, so it is never promoted to a lock
                if self.pend_n >= c.confirm_frames and (moved > c.move_deg or self.pend_dom):
                    trk.init(z)
                    trk.update(z, d.conf)
                    self.status, self.miss, self.acquired = LOCKED, 0, True
                    self.snr_ema = d.snr if self.snr_ema is None else 0.7 * self.snr_ema + 0.3 * d.snr
                    self.center = trk.pos()
                    self.pending, self.pend_n = None, 0
                    chosen = (z, d)
            else:
                self.pending, self.pend_n = None, 0
        # ---- control ----
        if self.status in (LOCKED, COASTING):
            aim = trk.predict_ahead(c.lead_frames * self.dt)
            v = trk.vel() + c.kp * (aim - enc_v)
        else:
            wp = self.levy.waypoint(self.center, enc_v, self.dt)
            v = c.search_kp * (wp - enc_v)
            n = np.linalg.norm(v)
            if n > c.search_v_max:
                v *= c.search_v_max / n
        pa = trk.pos() if trk.ready else self.center
        return v, Output(self.status, trk.model, chosen[1].conf if chosen else 0.0,
                         float(pa[0]), float(pa[1]), float(self.center[0]), float(self.center[1]),
                         len(dets), trk.ready)
