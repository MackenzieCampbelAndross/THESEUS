"""Closed-loop coarse alignment: detect -> track/predict -> status (LOCKED/COASTING/LOST) -> gimbal rates.
Lost-beacon recovery uses a prediction-anchored expanding spiral search."""
import math
from dataclasses import dataclass
import numpy as np
from .config import Config
from .detector import BeaconDetector
from .tracker import AdaptiveTracker

LOCKED, COASTING, LOST = "LOCKED", "COASTING", "LOST"


class SpiralSearch:
    """Deterministic Archimedean spiral anchored on the coarse cue. Used for INITIAL acquisition so a
    beacon that starts near/outside the FOV edge is swept into view in ~1 s, unlike a random walk that
    can stall. Re-acquisition after a loss still uses the prediction-guided LevySearch."""

    def __init__(self, c, dt):
        self.c, self.dt = c, dt
        self.reset()

    def reset(self):
        self.t = 0.0

    def waypoint(self, center, enc, dt):
        self.t += dt
        R = min(self.c.levy_r_max_deg, self.c.levy_r0_deg + self.c.levy_r_growth_deg_s * self.t)
        turn = 1.6                                   # rad/s sweep rate
        th = self.t * turn
        return center + R * np.array([math.cos(th), math.sin(th)])


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
        self.spiral = SpiralSearch(self.c, self.dt)
        self.drop = 0
        self.cx, self.cy, self.f = cx, cy, f_px
        self.status = LOST
        self.miss = 0
        self.pending = None
        self.pend_n = 0
        self.pend_hist = []
        self.pend_dom = True
        self.pend_age = 0
        self.center = np.array(cue, float)      # where to search (coarse cue, later the prediction)
        self.search_center = np.array(cue, float)  # frozen last-known position used as the sweep anchor
        self.snr_ema = None                     # brightness signature of the tracked beacon
        self.clutter = []                       # learned world-fixed light sources [az, el, hits]
        self.off_deg = 99.0                      # predicted beacon offset from boresight (deg), for FOV hysteresis

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
                self.spiral.reset()
                self.search_center = self.center.copy()   # freeze sweep anchor at last-known (prediction diverges)
                self.pending, self.pend_n, self.pend_age, self.drop = None, 0, 0, 0
        else:  # LOST / initial search: confirm a strong, self-consistent detection before locking
            if trk.ready:
                trk.coast_damp()
                self.center = trk.pos()
            cand = [(z, d) for z, d in zs if d.conf >= c.c_acquire and not self._is_clutter(z) and self._sig_ok(d)]
            if cand:
                self.drop = 0
                z, d = max(cand, key=lambda x: x[1].snr)
                if self.pending is not None and np.linalg.norm(z - self.pending) < c.confirm_gate_deg:
                    self.pend_n += 1
                else:
                    self.pend_n, self.pend_hist, self.pend_age = 1, [], 0
                self.pend_age += 1
                self.pending = z
                self.pend_hist = (self.pend_hist + [z.copy()])[-c.confirm_frames:]
                moved = np.linalg.norm(z - self.pend_hist[0]) if len(self.pend_hist) >= c.confirm_frames else 0.0
                # promote only on demonstrated motion in the reconstructed frame: a real beacon moves
                # (target flight), a static light does not (platform sway is far smaller over the window)
                if self.pend_n >= c.confirm_frames and moved > c.move_deg:
                    trk.init(z)
                    trk.update(z, d.conf)
                    self.status, self.miss = LOCKED, 0
                    self.snr_ema = d.snr if self.snr_ema is None else 0.7 * self.snr_ema + 0.3 * d.snr
                    self.center = trk.pos()
                    self.pending, self.pend_n = None, 0
                    chosen = (z, d)
                elif self.pend_age > 3 * c.confirm_frames:
                    self.pending, self.pend_n, self.pend_age = None, 0, 0   # candidate never moved (clutter): resume search
            elif self.pending is not None and self.drop < 2:
                self.drop += 1          # tolerate a brief dropout without resetting confirmation
            else:
                self.pending, self.pend_n, self.drop = None, 0, 0
        # ---- control ----
        if self.status in (LOCKED, COASTING):
            aim = trk.predict_ahead(c.lead_frames * self.dt)
            v = trk.vel() + c.kp * (aim - enc_v)
            n = np.linalg.norm(v)
            if n > c.track_v_max:                       # prevent fling-away on a bad maneuver velocity spike
                v *= c.track_v_max / n
        else:
            if self.pending is not None:
                wp = self.pending                       # hold on a candidate so confirmation completes
            else:
                wp = self.spiral.waypoint(self.search_center, enc_v, self.dt)  # sweep around frozen last-known
            v = c.search_kp * (wp - enc_v)
            n = np.linalg.norm(v)
            if n > c.search_v_max:
                v *= c.search_v_max / n
        pa = trk.pos() if trk.ready else self.center
        self.off_deg = float(np.linalg.norm(self.center - enc_v))
        return v, Output(self.status, trk.model, chosen[1].conf if chosen else 0.0,
                         float(pa[0]), float(pa[1]), float(self.center[0]), float(self.center[1]),
                         len(dets), trk.ready)
