"""Self-Adaptive Predictive Tracking Loop (SAPTL) estimator.

Two Kalman filters run in parallel on the beacon's absolute line-of-sight angle (az, el):
  * CV : constant-velocity model            (straight-line motion)
  * CT : constant-turn-rate model, omega estimated online from the heading change of the
         velocity vector                    (curved motion)
Each filter's innovation likelihood is accumulated with exponential forgetting; the tracker
switches to the model that explains the measurements better (with hysteresis), so the choice is
data-driven and not scripted."""
import math
import numpy as np
from .config import TrackerCfg

H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


def f_cv(dt):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def f_ct(dt, w):
    if abs(w) < 1e-4:
        return f_cv(dt)
    s, c = math.sin(w * dt), math.cos(w * dt)
    return np.array([[1, 0, s / w, -(1 - c) / w],
                     [0, 1, (1 - c) / w, s / w],
                     [0, 0, c, -s],
                     [0, 0, s, c]])


def q_mat(dt, q):
    g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    return g @ g.T * q ** 2


class KF:
    def __init__(self, name, q):
        self.name, self.q = name, q
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.ll = 0.0

    def init(self, z, p0=0.15, v0=25.0):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([p0 ** 2, p0 ** 2, v0 ** 2, v0 ** 2])
        self.ll = 0.0

    def F(self, dt, w):
        return f_cv(dt) if self.name == "CV" else f_ct(dt, w)

    def predict(self, dt, w):
        F = self.F(dt, w)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q_mat(dt, self.q)

    def update(self, z, R, forget):
        S = H @ self.P @ H.T + R
        nu = z - H @ self.x
        Si = np.linalg.inv(S)
        K = self.P @ H.T @ Si
        self.x = self.x + K @ nu
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        ll = -0.5 * (nu @ Si @ nu + math.log(np.linalg.det(S)))
        self.ll = forget * self.ll + ll


class AdaptiveTracker:
    def __init__(self, cfg: TrackerCfg, px_per_deg, dt):
        self.c, self.dt, self.ppd = cfg, dt, px_per_deg
        self.cv, self.ct = KF("CV", cfg.q_cv), KF("CT", cfg.q_ct)
        self.model = "CV"
        self.omega = 0.0            # rad/s, estimated
        self.heading = None
        self.ready = False
        self.model_history = []

    @property
    def sel(self):
        return self.ct if self.model == "CT" else self.cv

    def init(self, z):
        self.cv.init(z)
        self.ct.init(z)
        self.model, self.omega, self.heading, self.ready = "CV", 0.0, None, True

    def R(self, conf):
        s = self.c.sigma_px / self.ppd * (1 + self.c.conf_inflate * (1 - conf))
        return np.eye(2) * s ** 2

    def predict(self):
        w = self.omega
        self.cv.predict(self.dt, w)
        self.ct.predict(self.dt, w)

    def coast_damp(self):
        for f in (self.cv, self.ct):
            f.x[2:] *= self.c.coast_damp
            sp = math.hypot(f.x[2], f.x[3])
            if sp > 40.0:                      # physical sanity limit on line-of-sight rate (deg/s)
                f.x[2:] *= 40.0 / sp

    def pos(self):
        return self.sel.x[:2].copy()

    def vel(self):
        return self.sel.x[2:].copy()

    def pos_sigma(self):
        return math.sqrt(max(np.trace(self.sel.P[:2, :2]) / 2, 0.0))

    def update(self, z, conf):
        R = self.R(conf)
        self.cv.update(z, R, self.c.ll_forget)
        self.ct.update(z, R, self.c.ll_forget)
        self._adapt()

    def _adapt(self):
        c = self.c
        vx, vy = self.cv.x[2], self.cv.x[3]
        if math.hypot(vx, vy) > 1.0:                 # deg/s
            h = math.atan2(vy, vx)
            if self.heading is not None:
                d = (h - self.heading + math.pi) % (2 * math.pi) - math.pi
                w = d / self.dt
                self.omega = (1 - c.omega_alpha) * self.omega + c.omega_alpha * float(np.clip(w, -1.5, 1.5))
            self.heading = h
        big_turn = abs(math.degrees(self.omega)) > c.omega_min
        if self.model == "CV" and big_turn and self.ct.ll - self.cv.ll > c.switch_on:
            self.model = "CT"
        elif self.model == "CT" and (not big_turn or self.cv.ll - self.ct.ll > c.switch_off):
            self.model = "CV"

    def predict_ahead(self, horizon_s):
        """Propagate the selected model `horizon_s` ahead without touching filter state."""
        n = max(1, int(round(horizon_s / self.dt)))
        F = self.sel.F(horizon_s if n == 1 else horizon_s, self.omega)
        return (F @ self.sel.x)[:2]
