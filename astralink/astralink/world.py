"""Virtual FSOC scene: moving terminal (beacon), mobile platform disturbances, atmosphere,
weather, virtual 640x480 camera and a gimbal. Ground truth stays inside this module and is only
exposed through `truth` dictionaries used for scoring - never fed to the tracking pipeline."""
import math
import numpy as np
from .config import Config, WEATHER


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


class Trajectory:
    """3-D flight path of Terminal 2 (ENU, metres) built from a maneuver schedule."""

    def __init__(self, cfg: Config, rng):
        s, dt = cfg.scenario, 1.0 / cfg.camera.fps
        n = int(s.duration_s / dt) + 3
        self.dt, self.n = dt, n
        self.pos = np.zeros((n, 3))
        self.omega = np.zeros(n)            # true yaw rate (rad/s), for analysis only
        r0 = rng.uniform(*s.start_range_m)
        b = rng.uniform(0, 2 * math.pi)
        p = np.array([r0 * math.sin(b), r0 * math.cos(b), rng.uniform(*s.start_alt_m)])
        speed = s.speed_mps * (1.6 if s.kind == "aggressive" else 1.0)
        if s.kind == "straight":            # fly past the terminal with a lateral offset
            aim = math.atan2(-p[0] + rng.uniform(-70, 70), -p[1] + rng.uniform(-70, 70))
            heading = aim
        else:
            heading = rng.uniform(0, 2 * math.pi)
        climb = 0.0
        seg_left, om = 0.0, 0.0
        for i in range(n):
            self.pos[i] = p
            rng_h = math.hypot(p[0], p[1])
            if seg_left <= 0:
                seg_left = rng.uniform(2.0, 6.0)
                if s.kind == "straight":
                    om, seg_left = 0.0, 1e9
                elif s.kind == "circular":
                    om, seg_left = 0.22 * rng.choice([-1, 1]), 1e9
                else:
                    wmax = 0.8 if s.kind == "aggressive" else 0.45
                    om = 0.0 if rng.random() < 0.35 else rng.uniform(0.12, wmax) * rng.choice([-1, 1])
                    toward = math.atan2(-p[0], -p[1])
                    diff = wrap180(math.degrees(toward - heading))
                    if rng_h > s.max_range_m:
                        om = abs(om) * (1 if diff > 0 else -1)   # turn back toward the terminal
                    elif rng_h < s.min_range_m:
                        om = abs(om) * (-1 if diff > 0 else 1)   # turn away
                climb = rng.uniform(-3.0, 3.0)
            if p[2] < 15 and climb < 0 or p[2] > 200 and climb > 0:
                climb = -climb
            self.omega[i] = om
            heading += om * dt
            seg_left -= dt
            p = p + dt * np.array([speed * math.sin(heading), speed * math.cos(heading), climb])

    def los(self, k):
        e, n_, u = self.pos[min(k, self.n - 1)]
        h = math.hypot(e, n_)
        return math.degrees(math.atan2(e, n_)), math.degrees(math.atan2(u, h)), math.sqrt(e * e + n_ * n_ + u * u)


class Platform:
    """Terminal-1 attitude disturbance: slow sway + multi-tone vibration + white jitter (deg)."""

    def __init__(self, cfg: Config, rng, px_per_deg, vib_px):
        self.rng = rng
        self.sway_a = rng.uniform(0.6, 1.0, 2) * cfg.scenario.sway_deg
        self.sway_f = rng.uniform(0.05, 0.15, 2)
        self.sway_p = rng.uniform(0, 2 * math.pi, 2)
        self.vf = rng.uniform(2.0, 13.0, (2, 3))
        self.vp = rng.uniform(0, 2 * math.pi, (2, 3))
        self.vamp = vib_px / px_per_deg * 0.75 / math.sqrt(1.5)
        self.white = vib_px / px_per_deg * 0.35

    def att(self, t):
        out = []
        for a in range(2):
            sway = self.sway_a[a] * math.sin(2 * math.pi * self.sway_f[a] * t + self.sway_p[a])
            vib = self.vamp * np.sum(np.sin(2 * math.pi * self.vf[a] * t + self.vp[a]))
            out.append(sway + vib + self.rng.normal(0, self.white))
        return out[0], out[1]


class Atmosphere:
    """AR(1) scintillation (log-normal intensity) and beam wander (centroid jitter)."""

    def __init__(self, rng, scint, wander, rho=0.85):
        self.rng, self.scint, self.wander, self.rho = rng, scint, wander, rho
        self.g, self.w = 0.0, np.zeros(2)

    def step(self):
        c = math.sqrt(1 - self.rho ** 2)
        self.g = self.rho * self.g + c * self.rng.normal()
        self.w = self.rho * self.w + c * self.rng.normal(size=2)
        return math.exp(self.scint * self.g - self.scint ** 2 / 2), self.w * self.wander


class Occlusions:
    """Random temporary beacon blockage (cloud / obstruction / fade) - a Poisson process."""

    def __init__(self, cfg: Config, rng):
        s = cfg.scenario
        self.iv, t = [], rng.exponential(60.0 / max(s.occlusion_rate_per_min, 1e-9)) + 4.0
        while s.occlusion_rate_per_min > 0 and t < s.duration_s:
            d = rng.uniform(*s.occlusion_dur_s)
            self.iv.append((t, t + d))
            t += d + rng.exponential(60.0 / s.occlusion_rate_per_min)

    def factor(self, t):
        f = 1.0
        for a, b in self.iv:
            if a <= t <= b:
                return 0.0
            if a - 0.1 <= t < a:
                f = min(f, (a - t) / 0.1)
            elif b < t <= b + 0.1:
                f = min(f, (t - b) / 0.1)
        return f


class Gimbal:
    """Two-axis pan/tilt with a velocity loop, accel/rate limits and one-frame command latency."""

    def __init__(self, cfg: Config, az0, el0, rng):
        self.c, self.rng = cfg.gimbal, rng
        self.ang = np.array([az0, el0], float)
        self.vel = np.zeros(2)
        self.pending = np.zeros(2)

    def read(self):
        return tuple(self.ang + self.rng.normal(0, self.c.encoder_noise_deg, 2))

    def command(self, v_az, v_el):
        self.pending = np.clip([v_az, v_el], -self.c.v_max, self.c.v_max)

    def step(self, dt):
        dv = np.clip(self.c.kv * (self.pending - self.vel) * dt, -self.c.a_max * dt, self.c.a_max * dt)
        self.vel = np.clip(self.vel + dv, -self.c.v_max, self.c.v_max)
        self.ang += self.vel * dt
        self.ang[1] = np.clip(self.ang[1], self.c.el_min, self.c.el_max)


def add_spot(img, u, v, amp, sx, sy):
    h, w = img.shape
    rx, ry = int(math.ceil(5 * sx)) + 1, int(math.ceil(5 * sy)) + 1
    x0, x1 = int(u) - rx, int(u) + rx + 1
    y0, y1 = int(v) - ry, int(v) + ry + 1
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    xs = np.arange(x0, x1) - u
    ys = np.arange(y0, y1) - v
    img[y0:y1, x0:x1] += amp * np.outer(np.exp(-ys ** 2 / (2 * sy ** 2)), np.exp(-xs ** 2 / (2 * sx ** 2)))


class VirtualScene:
    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        r = self.rng
        cam, sc = cfg.camera, cfg.scenario
        self.W, self.H = cam.width, cam.height
        self.cx, self.cy = (self.W - 1) / 2, (self.H - 1) / 2
        self.f = (self.W / 2) / math.tan(math.radians(cam.hfov_deg / 2))
        self.px_per_deg = self.f * math.radians(1.0)
        wx = WEATHER[sc.weather]
        self.wx = wx
        self.alpha = 3.912 / wx["vis_km"] / 1000.0                       # extinction per metre (Koschmieder)
        self.traj = Trajectory(cfg, r)
        self.platform = Platform(cfg, r, self.px_per_deg, wx["vib"])
        self.atm = Atmosphere(r, wx["scint"], wx["wander"])
        self.occ = Occlusions(cfg, r)
        az0, el0, _ = self.traj.los(0)
        self.cue = (az0 + r.normal(0, sc.cue_sigma_deg), el0 + r.normal(0, sc.cue_sigma_deg * 0.6))
        # fixed world-frame clutter: point lights + a broad glow (not tied to the beacon)
        self.dist = [(az0 + r.uniform(-25, 25), el0 + r.uniform(-18, 18), r.uniform(12, 45))
                     for _ in range(wx["n_distract"])]
        self.glare = (az0 + r.uniform(-15, 15), el0 + r.uniform(-8, 12), r.uniform(25, 45))
        self.hot = (r.integers(0, self.H, cam.n_hot_pixels), r.integers(0, self.W, cam.n_hot_pixels),
                    r.uniform(60, 200, cam.n_hot_pixels))
        yy = np.linspace(0, 1, self.H, dtype=np.float32)[:, None]
        self.base = (12 + 25 * (1 - yy) + wx["veil"]) * np.ones((1, self.W), np.float32)
        self.gx = np.arange(self.W, dtype=np.float32)
        self.gy = np.arange(self.H, dtype=np.float32)
        self.prev_uv = None

    def dark_frame(self, n=8):
        """Shutter-closed frames (sensor defects + read noise only) used for fixed-pattern calibration."""
        acc = np.zeros((self.H, self.W), np.float32)
        for _ in range(n):
            img = np.zeros((self.H, self.W), np.float32)
            img[self.hot[0], self.hot[1]] += self.hot[2]
            img += self.cfg.camera.read_noise * self.rng.standard_normal(img.shape, dtype=np.float32)
            acc += img
        return acc / n

    def proj(self, daz, dele):
        return (self.cx + self.f * math.tan(math.radians(daz)),
                self.cy - self.f * math.tan(math.radians(dele)))

    def capture(self, k, gimbal_true):
        """Render frame k for the given *true* gimbal pose. Returns (uint8 frame, truth dict)."""
        cam, r = self.cfg.camera, self.rng
        t = k / cam.fps
        az, el, rng_m = self.traj.los(k)
        patt, patt_e = self.platform.att(t)
        gi, wander = self.atm.step()
        occ = self.occ.factor(t)
        # physical line-of-sight offset of the beacon from the optical axis (what truly matters)
        daz = wrap180(az - gimbal_true[0] - patt)
        dele = el - gimbal_true[1] - patt_e
        u, v = self.proj(daz, dele)
        visible = (0 <= u < self.W) and (0 <= v < self.H)
        T = math.exp(-self.alpha * rng_m)
        amp = cam.beacon_ref_amp * (cam.beacon_ref_range_m / rng_m) ** 2 * T * gi * occ
        img = self.base.copy()
        # glow
        gu, gv = self.proj(wrap180(self.glare[0] - gimbal_true[0] - patt), self.glare[1] - gimbal_true[1] - patt_e)
        img += self.glare[2] * np.outer(np.exp(-(self.gy - gv) ** 2 / (2 * 70.0 ** 2)),
                                        np.exp(-(self.gx - gu) ** 2 / (2 * 90.0 ** 2))).astype(np.float32)
        for (da, de, a) in self.dist:     # background point lights fixed in the world
            du, dv = self.proj(wrap180(da - gimbal_true[0] - patt), de - gimbal_true[1] - patt_e)
            add_spot(img, du, dv, a * (0.6 + 0.4 * T), 1.5, 1.5)
        # beacon: defocus/scatter + turbulence broadening + motion blur (anisotropic)
        vel = (0.0, 0.0) if self.prev_uv is None else (u - self.prev_uv[0], v - self.prev_uv[1])
        self.prev_uv = (u, v)
        s0 = cam.spot_sigma_px * (1 + 0.6 * (1 - T) + 0.3 * self.wx["scint"])
        bl = cam.exposure_frac
        sx = math.sqrt(s0 ** 2 + (vel[0] * bl) ** 2 / 12)
        sy = math.sqrt(s0 ** 2 + (vel[1] * bl) ** 2 / 12)
        ub, vb = u + wander[0], v + wander[1]
        if amp > 0.05:
            add_spot(img, ub, vb, amp, sx, sy)
            add_spot(img, ub, vb, 0.12 * (1 - T) * amp, 6 * s0, 6 * s0)      # forward-scatter halo
        img[self.hot[0], self.hot[1]] += self.hot[2]
        img += cam.shot_noise_k * np.sqrt(np.maximum(img, 0)) * r.standard_normal(img.shape, dtype=np.float32)
        img += cam.read_noise * r.standard_normal(img.shape, dtype=np.float32)
        frame = np.clip(img, 0, 255).astype(np.uint8)
        truth = dict(u=u, v=v, visible=visible, range_m=rng_m, amp=amp, az=az, el=el,
                     omega=self.traj.omega[min(k, self.traj.n - 1)], occluded=occ < 0.5,
                     speed_px=math.hypot(*vel))
        return frame, truth
