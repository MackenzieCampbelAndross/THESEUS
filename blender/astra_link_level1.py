# ============================================================================
# ASTRA LINK - 30 SECOND JURY DEMO  (V7, FIXED)
# SIH26169 | Virtual Camera Tracking for Coarse Alignment of Mobile FSOC
# Blender 4.2+ / 5.x
#
# HOW TO RUN
#   1. Open a new Blender file. Scripting workspace -> Text Editor -> open this file.
#   2. Alt+P (Run Script). Wait for the READY banner in the system console.
#   3. Press Home in the viewport, then SPACE from frame 1.
#   4. Render: Render > Render Animation (same Blender session, handler must be alive).
#
# TIMELINE
#   0-10s    Both terminals patrol. Side view.
#   10-12s   T1 nose camera scans, detects T2 beacon, acquires, locks (<= 2 s).
#   13-19s   T2 flies a corkscrew maneuver. Tracker switches CV -> CT.
#   16-17s   Simulated turbulence dropout: COASTING -> LOST -> REACQUIRING -> LOCKED.
#   17.6-18.5s Alignment verification + secure session (TLS 1.3 demonstrator).
#   18.9-30s External view. Laser link. Selective Repeat ARQ transfer. 100% at ~29 s.
#
# Everything above the "SCENE BUILD" marker is pure Python and bpy free.
# ============================================================================
import math
import random
from mathutils import Vector, Matrix, Euler

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FPS = 30
DURATION = 30.0
END_FRAME = int(FPS * DURATION)
SEED = 26169

# Virtual 640x480 camera. 9 deg vertical, 4:3 window -> about 12 x 9 deg.
LOGICAL_W = 640.0
LOGICAL_H = 480.0
FOV_Y = math.radians(9.0)
TAN_Y = math.tan(FOV_Y / 2.0)
TAN_X = TAN_Y * LOGICAL_W / LOGICAL_H
K_X = 2.0 * TAN_X          # normalized image units -> tangent of angle (x)
K_Y = 2.0 * TAN_Y          # normalized image units -> tangent of angle (y)

# Gimbal
PAN_MIN, PAN_MAX = math.radians(-70.0), math.radians(70.0)
TILT_MIN, TILT_MAX = math.radians(-35.0), math.radians(35.0)
GIMBAL_RATE = math.radians(20.0)
SEARCH_RATE = math.radians(10.0)
PATROL_PAN = math.radians(-6.0)
PATROL_TILT = math.radians(-3.5)
SEARCH_PAN_MIN = math.radians(-10.0)
SEARCH_PAN_MAX = math.radians(36.0)
SEARCH_TILT_C = math.radians(-3.5)
SEARCH_TILT_A = math.radians(1.5)

# Timeline
SEARCH_START = 10.0
MANEUVER_START = 13.0
MANEUVER_R = 6.0
MANEUVER_W = 0.5
DROPOUT = (16.0, 16.75)
VERIFY_START = 17.6
VERIFY_DURATION = 0.9
CUT_EXTERNAL = 18.9
LINK_START = 18.9
TRANSFER_START = 20.0

# Tracker thresholds
LOCK_CONF = 0.58
MEAS_CONF = 0.32
LOCK_HITS = 7
REACQ_HITS = 5
COAST_TIME = 0.45

# Geometry of the T1 nose gimbal (local frames, metres)
NOSE_X = 5.05
NOSE_Z = -0.12
TILT_OFF = Vector((0.28, 0.0, -0.16))
CAM_OFF = Vector((1.60, 0.0, 0.0))
LENS_OFF = Vector((1.54, 0.0, 0.0))
BEACON_LOCAL = Vector((4.12, 0.0, 1.60))
CAMERA_BASE_Q = Vector((1.0, 0.0, 0.0)).to_track_quat('-Z', 'Y')

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def step_toward(cur, target, rate, dt):
    return cur + clamp(target - cur, -rate * dt, rate * dt)


def frame_rng(frame, salt=0):
    return random.Random(SEED + int(frame) * 7919 + salt)


def levy_sample(rng, beta=1.5):
    u = rng.gauss(0.0, 1.0)
    v = abs(rng.gauss(0.0, 1.0)) + 1e-6
    return clamp(u / (v ** (1.0 / beta)), -4.0, 4.0)


# ----------------------------------------------------------------------------
# Terminal motion (world frame, metres)
# ----------------------------------------------------------------------------
def pos1(t):
    return Vector((
        -85.0 + 3.0 * t,
        -40.0 + 4.0 * math.sin(0.15 * t),
        70.0 + 2.0 * math.sin(0.20 * t),
    ))


def _smooth(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pos2(t):
    p = Vector((
        -5.0 + 3.0 * t,
        -20.0 + 3.0 * math.sin(0.18 * t),
        73.0 + 1.0 * math.sin(0.22 * t),
    ))
    u = t - MANEUVER_START
    if u > 0.0:
        e = _smooth(u / 1.2)
        w = MANEUVER_W
        r = MANEUVER_R
        # corkscrew: velocity vector in the lateral/vertical plane rotates at w rad/s
        p += Vector((0.0, e * r * math.sin(w * u), e * r * (1.0 - math.cos(w * u))))
    return p


def attitude(fn, t, pitch_gain, roll_amp, roll_w):
    h = 0.08
    v = (fn(t + h) - fn(t - h)) / (2.0 * h)
    yaw = math.atan2(v.y, v.x)
    pitch = math.atan2(v.z, math.hypot(v.x, v.y))
    return Euler((roll_amp * math.sin(roll_w * t), -pitch * pitch_gain, yaw), 'XYZ')


def scenario_profile(t):
    if t < MANEUVER_START:
        return dict(noise_px=0.6, fog=0.0, jitter_deg=0.0, blocked=False)
    if t < 19.0:
        blocked = DROPOUT[0] <= t < DROPOUT[1]
        return dict(noise_px=1.2, fog=0.10, jitter_deg=0.03, blocked=blocked)
    return dict(noise_px=0.8, fog=0.04, jitter_deg=0.015, blocked=False)


# ----------------------------------------------------------------------------
# Analytic kinematics. Mirrors the Blender parent chain exactly:
#   T1 -> GIMBAL_BASE -> PAN(z) -> TILT(y) -> CAMERA
# ----------------------------------------------------------------------------
def camera_quat(frame, profile):
    j = math.radians(profile['jitter_deg'])
    if j <= 0.0:
        return CAMERA_BASE_Q.copy()
    r = frame_rng(frame, 310)
    jq = Euler((0.0, r.gauss(0.0, 1.0) * j, r.gauss(0.0, 1.0) * j), 'XYZ').to_quaternion()
    return jq @ CAMERA_BASE_Q


def gimbal_chain(p1, e1, pan, tilt, cam_q):
    m = Matrix.Translation(p1) @ e1.to_matrix().to_4x4()
    m = m @ Matrix.Translation(Vector((NOSE_X, 0.0, NOSE_Z)))
    m = m @ Matrix.Rotation(pan, 4, 'Z')
    m = m @ Matrix.Translation(TILT_OFF)
    m = m @ Matrix.Rotation(tilt, 4, 'Y')
    tilt_world = m
    cam_world = m @ Matrix.Translation(CAM_OFF) @ cam_q.to_matrix().to_4x4()
    return cam_world, tilt_world


def beacon_world(p2, e2):
    m = Matrix.Translation(p2) @ e2.to_matrix().to_4x4()
    return m @ BEACON_LOCAL


def project(cam_world, point):
    local = cam_world.inverted() @ point
    f = -local.z
    if f <= 0.1:
        return None
    u = 0.5 + (local.x / f) / (2.0 * TAN_X)
    v = 0.5 + (local.y / f) / (2.0 * TAN_Y)
    return u, v, f


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
class Detection:
    def __init__(self, measurement=None, confidence=0.0, valid=False, in_fov=False):
        self.measurement = measurement
        self.confidence = confidence
        self.valid = valid
        self.in_fov = in_fov


def run_detector(frame, truth, profile):
    if truth is None:
        return Detection()
    if profile['blocked']:
        return Detection(None, 0.10, False, True)
    u, v = truth.x, truth.y
    edge = min(u, 1.0 - u, v, 1.0 - v)
    conf = 0.99 * clamp(edge / 0.22, 0.0, 1.0) * (1.0 - 0.6 * profile['fog'])
    conf = clamp(conf - profile['noise_px'] / 22.0, 0.0, 0.99)
    r = frame_rng(frame, 911)
    z = Vector((
        clamp(u + r.gauss(0.0, profile['noise_px'] / LOGICAL_W), 0.0, 1.0),
        clamp(v + r.gauss(0.0, profile['noise_px'] / LOGICAL_H), 0.0, 1.0),
    ))
    valid = conf >= MEAS_CONF
    return Detection(z if valid else None, conf, valid, True)


# ----------------------------------------------------------------------------
# Kalman filter (per axis) + adaptive CV / CT motion model
# State velocity is the ABSOLUTE target rate in image units / s.
# Gimbal motion is removed in predict() (ego-motion compensation).
# ----------------------------------------------------------------------------
class AxisKF:
    def __init__(self):
        self.reset(0.5)

    def reset(self, p):
        self.p = p
        self.v = 0.0
        self.P00 = 0.02
        self.P01 = 0.0
        self.P11 = 0.05

    def predict(self, dt, q, ego, new_p=None, new_v=None):
        if new_p is None:
            self.p = self.p + self.v * dt + ego
        else:
            self.p = new_p + ego
            self.v = new_v
        P00 = self.P00 + dt * (2.0 * self.P01 + dt * self.P11) + q * dt ** 4 / 4.0
        P01 = self.P01 + dt * self.P11 + q * dt ** 3 / 2.0
        P11 = self.P11 + q * dt ** 2
        self.P00, self.P01, self.P11 = P00, P01, P11

    def update(self, z, r):
        s = self.P00 + r
        k0 = self.P00 / s
        k1 = self.P01 / s
        y = z - self.p
        self.p += k0 * y
        self.v += k1 * y
        P00 = (1.0 - k0) * self.P00
        P01 = (1.0 - k0) * self.P01
        P11 = self.P11 - k1 * self.P01
        self.P00, self.P01, self.P11 = P00, P01, max(P11, 1e-9)


class AdaptiveTracker:
    Q_CV = 0.15
    Q_CT = 0.12
    SPEED_MIN = 0.012      # rad/s of image-plane motion needed to judge curvature
    DWELL = 9              # frames a model change must persist

    def __init__(self):
        self.kx = AxisKF()
        self.ky = AxisKF()
        self.initialized = False
        self.model = 'CV'
        self.omega = 0.0
        self.last_heading = None
        self.ct_count = 0
        self.cv_count = 0

    def reset(self, z):
        self.kx.reset(z.x)
        self.ky.reset(z.y)
        self.initialized = True
        self.model = 'CV'
        self.omega = 0.0
        self.last_heading = None
        self.ct_count = 0
        self.cv_count = 0

    @property
    def position(self):
        return Vector((self.kx.p, self.ky.p))

    @property
    def velocity(self):
        return Vector((self.kx.v, self.ky.v))

    def observe_motion(self, dt):
        ax = self.kx.v * K_X
        ay = self.ky.v * K_Y
        if math.hypot(ax, ay) < self.SPEED_MIN:
            self.omega *= 0.92
            self.last_heading = None
        else:
            h = math.atan2(ay, ax)
            if self.last_heading is not None:
                raw = clamp(wrap(h - self.last_heading) / dt, -2.0, 2.0)
                self.omega = 0.90 * self.omega + 0.10 * raw
            self.last_heading = h
        if abs(self.omega) > 0.20:
            self.ct_count += 1
            self.cv_count = 0
        elif abs(self.omega) < 0.10:
            self.cv_count += 1
            self.ct_count = 0
        if self.model == 'CV' and self.ct_count >= self.DWELL:
            self.model = 'CT'
        elif self.model == 'CT' and self.cv_count >= self.DWELL:
            self.model = 'CV'

    def predict(self, dt, ego_u, ego_v):
        if not self.initialized:
            return
        if self.model == 'CT' and abs(self.omega) > 0.05:
            w = clamp(self.omega, -1.5, 1.5)
            ax = self.kx.v * K_X
            ay = self.ky.v * K_Y
            s = math.sin(w * dt)
            c = math.cos(w * dt)
            dx = (ax * s + ay * (c - 1.0)) / w
            dy = (ax * (1.0 - c) + ay * s) / w
            vx = ax * c - ay * s
            vy = ax * s + ay * c
            self.kx.predict(dt, self.Q_CT, ego_u, self.kx.p + dx / K_X, vx / K_X)
            self.ky.predict(dt, self.Q_CT, ego_v, self.ky.p + dy / K_Y, vy / K_Y)
        else:
            self.kx.predict(dt, self.Q_CV, ego_u)
            self.ky.predict(dt, self.Q_CV, ego_v)

    def update(self, z, confidence):
        if not self.initialized:
            self.reset(z)
            return
        r = (1.5 / LOGICAL_W) ** 2 * (1.0 + 8.0 * (1.0 - confidence))
        rv = (1.5 / LOGICAL_H) ** 2 * (1.0 + 8.0 * (1.0 - confidence))
        self.kx.update(z.x, r)
        self.ky.update(z.y, rv)


# ----------------------------------------------------------------------------
# PID + velocity feedforward. Pan is +LEFT, tilt is +DOWN.
# ----------------------------------------------------------------------------
class PIDVelocityFF:
    KP = 4.0
    KI = 1.0
    KD = 0.12
    KFF = 1.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.ix = 0.0
        self.iy = 0.0
        self.lx = None
        self.ly = None
        self.dx = 0.0
        self.dy = 0.0

    def compute(self, est, vel, dt):
        ex = -(est.x - 0.5) * K_X       # target right of centre -> pan right (negative)
        ey = -(est.y - 0.5) * K_Y       # target above centre    -> tilt up   (negative)
        self.ix = clamp(self.ix + ex * dt, -0.15, 0.15)
        self.iy = clamp(self.iy + ey * dt, -0.15, 0.15)
        if self.lx is not None:
            self.dx = 0.6 * self.dx + 0.4 * (ex - self.lx) / dt
            self.dy = 0.6 * self.dy + 0.4 * (ey - self.ly) / dt
        self.lx, self.ly = ex, ey
        pan = self.KP * ex + self.KI * self.ix + self.KD * self.dx + self.KFF * (-vel.x * K_X)
        tilt = self.KP * ey + self.KI * self.iy + self.KD * self.dy + self.KFF * (-vel.y * K_Y)
        return clamp(pan, -GIMBAL_RATE, GIMBAL_RATE), clamp(tilt, -GIMBAL_RATE, GIMBAL_RATE)


# ----------------------------------------------------------------------------
# Selective Repeat ARQ demonstrator (800 x 2.5 MB = 2 GB)
# ----------------------------------------------------------------------------
class SelectiveRepeatDemo:
    TOTAL = 800
    CHUNK_MB = 2.5
    SEND_PER_FRAME = 3
    TIMEOUT = 0.15
    LOSS = 0.03

    def __init__(self):
        self.reset()

    def reset(self):
        self.next_seq = 0
        self.acked = set()
        self.pending = {}
        self.retx = 0
        self.sent = 0

    def step(self, frame, t):
        r = frame_rng(frame, 4401)
        budget = self.SEND_PER_FRAME
        while budget > 0 and self.next_seq < self.TOTAL:
            seq = self.next_seq
            self.next_seq += 1
            self.sent += 1
            budget -= 1
            if r.random() < self.LOSS:
                self.pending[seq] = t
            else:
                self.acked.add(seq)
        due = sorted(s for s, ts in self.pending.items() if t - ts >= self.TIMEOUT)[:8]
        for seq in due:
            self.retx += 1
            self.sent += 1
            if r.random() >= self.LOSS:
                self.acked.add(seq)
                del self.pending[seq]
            else:
                self.pending[seq] = t
        SIM['transfer_progress'] = len(self.acked) / float(self.TOTAL)
        if len(self.acked) >= self.TOTAL:
            SIM['transfer_complete'] = True


# ----------------------------------------------------------------------------
# Simulation state
# ----------------------------------------------------------------------------
def new_state():
    z = Vector((0.0, 0.0, 0.0))
    return dict(
        frame=None, t=0.0, state='PATROLLING',
        confidence=0.0, measurement=None, prediction=None, truth=None,
        tracking_error_px=None, err_hist=[],
        lock_hits=0, last_detection=-99.0,
        coast_t0=None, lost_t0=None, loss_t0=None,
        acq_time=None, reacq_time=None,
        verify_t0=None, session_authenticated=False, link_established=False,
        transfer_complete=False, transfer_progress=0.0,
        pan=PATROL_PAN, tilt=PATROL_TILT, ego=(0.0, 0.0),
        pan_view=PATROL_PAN, tilt_view=PATROL_TILT,
        search_dir=1.0, levy=(0.0, 0.0), levy_frame=0,
        p1=z.copy(), p2=z.copy(), e1=Euler((0, 0, 0), 'XYZ'), e2=Euler((0, 0, 0), 'XYZ'),
        cam_q=CAMERA_BASE_Q.copy(), beacon=z.copy(), lens=z.copy(),
    )


SIM = new_state()
TRACKER = AdaptiveTracker()
CONTROLLER = PIDVelocityFF()
LINK = SelectiveRepeatDemo()


def reset_sim():
    SIM.clear()
    SIM.update(new_state())
    TRACKER.__init__()
    CONTROLLER.reset()
    LINK.reset()


def search_step(pan, tilt, t, dt):
    pan += SIM['search_dir'] * SEARCH_RATE * dt
    if pan >= SEARCH_PAN_MAX:
        pan = SEARCH_PAN_MAX
        SIM['search_dir'] = -1.0
    elif pan <= SEARCH_PAN_MIN:
        pan = SEARCH_PAN_MIN
        SIM['search_dir'] = 1.0
    tt = SEARCH_TILT_C + SEARCH_TILT_A * math.sin(1.3 * (t - SEARCH_START))
    return pan, step_toward(tilt, tt, GIMBAL_RATE, dt)


def reacq_step(frame, pan, tilt, dt):
    if SIM['levy_frame'] == 0 or frame % 9 == 0:
        r = frame_rng(frame, 7241)
        SIM['levy'] = (pan + math.radians(2.0) * levy_sample(r),
                       tilt + math.radians(0.9) * levy_sample(r))
        SIM['levy_frame'] = frame
    est = TRACKER.position
    pred_pan = pan - (est.x - 0.5) * K_X
    pred_tilt = tilt - (est.y - 0.5) * K_Y
    tp = clamp(0.7 * pred_pan + 0.3 * SIM['levy'][0], PAN_MIN, PAN_MAX)
    tt = clamp(0.7 * pred_tilt + 0.3 * SIM['levy'][1], TILT_MIN, TILT_MAX)
    return step_toward(pan, tp, GIMBAL_RATE, dt), step_toward(tilt, tt, GIMBAL_RATE, dt)


def sim_step(frame):
    S = SIM
    t = (frame - 1) / float(FPS)
    dt = 1.0 / FPS
    S['frame'] = frame
    S['t'] = t

    p1 = pos1(t)
    e1 = attitude(pos1, t, 0.3, math.radians(1.0), 0.22)
    p2 = pos2(t)
    e2 = attitude(pos2, t, 0.5, math.radians(1.0), 0.24)
    profile = scenario_profile(t)

    pan, tilt = S['pan'], S['tilt']          # gimbal pose seen by the camera this frame
    cam_q = camera_quat(frame, profile)
    cam_w, tilt_w = gimbal_chain(p1, e1, pan, tilt, cam_q)
    bw = beacon_world(p2, e2)
    proj = project(cam_w, bw)
    truth = None
    if proj is not None and 0.0 <= proj[0] <= 1.0 and 0.0 <= proj[1] <= 1.0:
        truth = Vector((proj[0], proj[1]))

    S.update(p1=p1, p2=p2, e1=e1, e2=e2, cam_q=cam_q, beacon=bw,
             lens=tilt_w @ LENS_OFF, pan_view=pan, tilt_view=tilt, truth=truth)

    if t < SEARCH_START:
        S['state'] = 'PATROLLING'
        S['measurement'] = None
        S['confidence'] = 0.0
        S['ego'] = (0.0, 0.0)
        return

    # ---- detection -------------------------------------------------------
    det = run_detector(frame, truth, profile)
    S['measurement'] = det.measurement
    S['confidence'] = det.confidence
    valid = det.valid and det.measurement is not None
    strong = valid and det.confidence >= LOCK_CONF

    if S['state'] == 'PATROLLING':
        S['state'] = 'SEARCHING'
        S['search_dir'] = 1.0
        TRACKER.__init__()
        CONTROLLER.reset()

    # ---- estimation ------------------------------------------------------
    if TRACKER.initialized:
        TRACKER.predict(dt, S['ego'][0], S['ego'][1])
    if valid:
        if not TRACKER.initialized:
            TRACKER.reset(det.measurement)
        else:
            TRACKER.update(det.measurement, det.confidence)
            TRACKER.observe_motion(dt)
        S['last_detection'] = t

    # ---- state machine ---------------------------------------------------
    st = S['state']
    if valid:
        if st == 'SEARCHING':
            st = 'ACQUIRING'
            S['lock_hits'] = 0
            CONTROLLER.reset()
        if st == 'ACQUIRING':
            S['lock_hits'] = S['lock_hits'] + 1 if strong else max(0, S['lock_hits'] - 1)
            if S['lock_hits'] >= LOCK_HITS:
                st = 'LOCKED'
                if S['acq_time'] is None:
                    S['acq_time'] = t - SEARCH_START
                CONTROLLER.reset()
        elif st == 'COASTING':
            if strong:
                st = 'LOCKED'
                S['reacq_time'] = t - S['loss_t0']
                CONTROLLER.reset()
        else:
            if st == 'LOST':
                st = 'REACQUIRING'
                S['lock_hits'] = 0
            if st == 'REACQUIRING':
                S['lock_hits'] = S['lock_hits'] + 1 if strong else 0
                if S['lock_hits'] >= REACQ_HITS:
                    st = 'LOCKED'
                    S['reacq_time'] = t - S['loss_t0']
                    S['lock_hits'] = 0
                    CONTROLLER.reset()
    else:
        if st == 'LOCKED':
            st = 'COASTING'
            S['coast_t0'] = t
            S['loss_t0'] = t
        elif st == 'COASTING' and t - S['coast_t0'] >= COAST_TIME:
            st = 'LOST'
            S['lost_t0'] = t
        elif st == 'LOST' and t - S['lost_t0'] >= 0.20:
            st = 'REACQUIRING'
            S['lock_hits'] = 0
            S['levy_frame'] = 0
        elif st == 'ACQUIRING' and t - S['last_detection'] > 0.5:
            st = 'SEARCHING'
            TRACKER.__init__()
            CONTROLLER.reset()
        elif st == 'REACQUIRING' and t - S['loss_t0'] > 6.0:
            st = 'SEARCHING'
            TRACKER.__init__()
            CONTROLLER.reset()
    S['state'] = st

    S['prediction'] = None
    if TRACKER.initialized:
        est = TRACKER.position
        S['prediction'] = Vector((clamp(est.x, 0.0, 1.0), clamp(est.y, 0.0, 1.0)))

    # ---- tracking error (10 frame mean, px) ------------------------------
    if truth is not None and TRACKER.initialized and st in ('ACQUIRING', 'LOCKED', 'COASTING'):
        est = TRACKER.position
        err = math.hypot((truth.x - est.x) * LOGICAL_W, (truth.y - est.y) * LOGICAL_H)
        S['err_hist'].append(err)
        S['err_hist'] = S['err_hist'][-10:]
        S['tracking_error_px'] = sum(S['err_hist']) / len(S['err_hist'])
    else:
        S['err_hist'] = []
        S['tracking_error_px'] = None

    # ---- gimbal command (applied to the NEXT frame) ----------------------
    new_pan, new_tilt = pan, tilt
    if st == 'SEARCHING':
        new_pan, new_tilt = search_step(pan, tilt, t, dt)
    elif st in ('ACQUIRING', 'LOCKED', 'COASTING') and TRACKER.initialized:
        pr, tr = CONTROLLER.compute(TRACKER.position, TRACKER.velocity, dt)
        new_pan = clamp(pan + pr * dt, PAN_MIN, PAN_MAX)
        new_tilt = clamp(tilt + tr * dt, TILT_MIN, TILT_MAX)
    elif st == 'REACQUIRING' and TRACKER.initialized:
        new_pan, new_tilt = reacq_step(frame, pan, tilt, dt)
    S['pan'], S['tilt'] = new_pan, new_tilt
    S['ego'] = ((new_pan - pan) / K_X, (new_tilt - tilt) / K_Y)

    # ---- alignment verification -> secure session -> link ---------------
    if not S['link_established']:
        err = S['tracking_error_px']
        ok = (t >= VERIFY_START and st == 'LOCKED' and err is not None
              and err <= 10.0 and S['confidence'] >= 0.70)
        if ok:
            if S['verify_t0'] is None:
                S['verify_t0'] = t
            if t - S['verify_t0'] >= VERIFY_DURATION:
                S['session_authenticated'] = True
                S['link_established'] = True
        else:
            S['verify_t0'] = None

    if S['link_established'] and t >= TRANSFER_START and not S['transfer_complete']:
        LINK.step(frame, t)


# ----------------------------------------------------------------------------
# HUD text content (pure strings, styles: norm / good / warn)
# ----------------------------------------------------------------------------
def _px(v):
    return '---' if v is None else '%03d,%03d' % (v.x * LOGICAL_W, (1.0 - v.y) * LOGICAL_H)


def stage_text():
    S = SIM
    t = S['t']
    s = S['state']
    if t < SEARCH_START:
        return 'PATROL | TERMINALS IN MOTION', 'norm'
    if s == 'SEARCHING':
        return 'COARSE SEARCH | SCANNING FOR T2 BEACON', 'warn'
    if s == 'ACQUIRING':
        return 'BEACON DETECTED | ACQUISITION', 'warn'
    if s in ('COASTING', 'LOST'):
        return 'SIGNAL DROPOUT | KALMAN PREDICTION HOLD', 'warn'
    if s == 'REACQUIRING':
        return 'REACQUISITION | PREDICTION GUIDED LEVY SEARCH', 'warn'
    if S['transfer_complete']:
        return 'TRANSFER COMPLETE | LINK HELD', 'good'
    if S['link_established'] and t >= TRANSFER_START:
        return 'FSOC LINK | OPTICAL DATA TX | SELECTIVE REPEAT ARQ', 'good'
    if S['link_established']:
        return 'ALIGNMENT VERIFIED | SECURE SESSION ESTABLISHED', 'good'
    if S['verify_t0'] is not None:
        return 'ALIGNMENT VERIFICATION | TLS 1.3 HANDSHAKE', 'norm'
    if t >= MANEUVER_START:
        return 'T2 MANEUVER | ADAPTIVE CV / CT TRACKING', 'norm'
    return 'LOCKED | SAPTL PREDICTIVE TRACKING', 'norm'


def hud_content():
    S = SIM
    t = S['t']
    st = S['state']
    warn = 'warn' if st in ('SEARCHING', 'ACQUIRING', 'COASTING', 'LOST', 'REACQUIRING') else 'norm'
    if t < SEARCH_START:
        det = 'STANDBY'
    elif S['measurement'] is not None:
        det = 'BEACON DETECTED'
    elif st in ('COASTING', 'LOST', 'REACQUIRING'):
        det = 'SIGNAL LOST'
    else:
        det = 'SCANNING'
    err = S['tracking_error_px']
    err_s = '---' if err is None else '%.1f' % err
    acq = '---' if S['acq_time'] is None else '%.2f S' % S['acq_time']
    rq = '---' if S['reacq_time'] is None else '%.2f S' % S['reacq_time']
    pan = math.degrees(S['pan_view'])
    tilt = math.degrees(S['tilt_view'])
    sess = 'AUTHENTICATED' if S['session_authenticated'] else (
        'VERIFYING' if S['verify_t0'] is not None else 'STANDBY')
    link = 'CONNECTED' if S['link_established'] else 'DISCONNECTED'
    good = 'good' if S['link_established'] else 'norm'
    txing = S['link_established'] and t >= TRANSFER_START
    tx = 'ACTIVE' if txing else 'STANDBY'
    prog = S['transfer_progress']
    n = int(prog * 20)
    bar = '#' * n + '-' * (20 - n)
    rate = LINK.CHUNK_MB * 8.0 * LINK.SEND_PER_FRAME * FPS / 1000.0
    rate_s = ('%.2f GBPS DEMO' % rate) if (txing and not S['transfer_complete']) else '0.00 GBPS'

    ext = [
        ('ASTRA LINK | VIRTUAL FSOC TESTBED', 'norm'),
        ('STATE: %s' % st, warn),
        ('DETECTOR: %s' % det, warn),
        ('MODEL: %s   CONF: %.2f' % (TRACKER.model, S['confidence']), 'norm'),
        ('CENTROID: %s   PRED: %s' % (_px(S['measurement']), _px(S['prediction'])), 'norm'),
        ('TRACK ERROR: %s PX (10F MEAN)' % err_s, 'norm'),
        ('PAN / TILT: %.1f / %.1f DEG' % (pan, tilt), 'norm'),
        ('ACQ: %s   REACQ: %s' % (acq, rq), 'norm'),
        ('PIPELINE: SAPTL|KALMAN|CV-CT|PID+FF', 'norm'),
        ('SESSION: %s' % sess, good),
        ('LINK: %s   TX: %s' % (link, tx), good),
        ('DATA [%s] %5.1f%%' % (bar, prog * 100.0), good),
        ('ARQ ACK %03d/%d RETX %02d PEND %02d  %s' % (
            len(LINK.acked), LINK.TOTAL, LINK.retx, len(LINK.pending), rate_s), good),
    ]
    ob = [
        ('ASTRA LINK | T1 NOSE CAMERA 640x480', 'norm'),
        ('STATE: %s' % st, warn),
        ('MODEL: %s   CONF: %.2f' % (TRACKER.model, S['confidence']), 'norm'),
        ('CENTROID: %s' % _px(S['measurement']), 'norm'),
        ('PREDICT : %s' % _px(S['prediction']), 'norm'),
        ('ERROR: %s PX' % err_s, 'norm'),
        ('PAN/TILT: %.1f / %.1f' % (pan, tilt), 'norm'),
        ('ACQ: %s   REACQ: %s' % (acq, rq), 'norm'),
        ('SESSION: %s' % sess, good),
    ]
    return ext, ob, stage_text()


# ##### SCENE BUILD (Blender only) #####
import bpy

scene = bpy.context.scene

# ----------------------------------------------------------------------------
# Cleanup of a previous run
# ----------------------------------------------------------------------------
for cb in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post):
    for h in list(cb):
        if getattr(h, '__name__', '').startswith('astra_link'):
            cb.remove(h)

try:
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
except Exception:
    pass

for obj in list(bpy.data.objects):
    if obj.name.startswith(('ASTRA_', 'TERMINAL_', 'T1_', 'T2_', 'FSOC_', 'OPEN_')):
        bpy.data.objects.remove(obj, do_unlink=True)

for cname in ['ASTRA_ENVIRONMENT', 'ASTRA_CITY', 'ASTRA_T1', 'ASTRA_T2',
              'ASTRA_CAMERAS', 'ASTRA_EFFECTS']:
    c = bpy.data.collections.get(cname)
    if c:
        try:
            bpy.data.collections.remove(c)
        except Exception:
            pass

# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------
def move_to_collection(obj, collection):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)


def mat(name, color, metallic=0.0, roughness=0.5, emission=None, emission_strength=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)
        bsdf.inputs['Metallic'].default_value = metallic
        bsdf.inputs['Roughness'].default_value = roughness
        if emission is not None:
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = (*emission, 1.0)
            if 'Emission Strength' in bsdf.inputs:
                bsdf.inputs['Emission Strength'].default_value = emission_strength
    m.diffuse_color = (*color, 1.0)
    return m


def add_cube(name, location, scale, material, collection, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(location=location)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material:
        o.data.materials.append(material)
    move_to_collection(o, collection)
    if bevel:
        b = o.modifiers.new('Bevel', 'BEVEL')
        b.width = bevel
        b.segments = 1
    return o


def add_uv(name, location, radius, material, collection, segments=12, rings=8):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings,
                                         radius=radius, location=location)
    o = bpy.context.object
    o.name = name
    if material:
        o.data.materials.append(material)
    move_to_collection(o, collection)
    return o


def add_cyl(name, location, radius, depth, material, collection, rotation=(0, 0, 0), vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth,
                                        location=location, rotation=rotation)
    o = bpy.context.object
    o.name = name
    if material:
        o.data.materials.append(material)
    move_to_collection(o, collection)
    return o


def add_torus(name, major, minor, material, collection, rotation=(0, 0, 0), segs=28):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor,
                                     major_segments=segs, minor_segments=6,
                                     location=(0, 0, 0), rotation=rotation)
    o = bpy.context.object
    o.name = name
    if material:
        o.data.materials.append(material)
    move_to_collection(o, collection)
    return o


def add_empty(name, location, collection, display='PLAIN_AXES'):
    o = bpy.data.objects.new(name, None)
    collection.objects.link(o)
    o.location = location
    o.empty_display_type = display
    o.empty_display_size = 1.0
    return o


def add_text(name, body, parent, location, size, material, collection, align='LEFT'):
    bpy.ops.object.text_add(location=(0, 0, 0))
    o = bpy.context.object
    o.name = name
    o.data.body = body
    o.data.align_x = align
    o.data.align_y = 'CENTER'
    o.data.size = size
    o.data.extrude = 0.0
    o.data.materials.append(material)
    o.parent = parent
    o.location = location
    move_to_collection(o, collection)
    try:
        o.visible_shadow = False
    except Exception:
        pass
    return o


def aim(obj, loc, target):
    obj.location = loc
    d = target - loc
    if d.length > 1e-6:
        obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


# ----------------------------------------------------------------------------
# Scene settings
# ----------------------------------------------------------------------------
scene.frame_start = 1
scene.frame_end = END_FRAME
scene.render.fps = FPS
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
for eng in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
    try:
        scene.render.engine = eng
        break
    except TypeError:
        continue
try:
    scene.render.image_settings.media_type = 'IMAGE'
except Exception:
    pass
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = '//ASTRA_JURY_RENDER/ASTRA_'

ASTRA_ENV = bpy.data.collections.new('ASTRA_ENVIRONMENT')
ASTRA_CITY = bpy.data.collections.new('ASTRA_CITY')
ASTRA_T1 = bpy.data.collections.new('ASTRA_T1')
ASTRA_T2 = bpy.data.collections.new('ASTRA_T2')
ASTRA_CAM = bpy.data.collections.new('ASTRA_CAMERAS')
ASTRA_FX = bpy.data.collections.new('ASTRA_EFFECTS')
for c in (ASTRA_ENV, ASTRA_CITY, ASTRA_T1, ASTRA_T2, ASTRA_CAM, ASTRA_FX):
    scene.collection.children.link(c)

# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------
GROUND = mat('ASTRA_Ground', (0.028, 0.036, 0.048), roughness=0.92)
ROAD = mat('ASTRA_Road', (0.010, 0.013, 0.017), roughness=0.90)
SIDEWALK = mat('ASTRA_Sidewalk', (0.10, 0.11, 0.12), roughness=0.94)
BUILD_DARK = mat('ASTRA_BuildDark', (0.028, 0.045, 0.064), metallic=0.42, roughness=0.38)
BUILD_LIGHT = mat('ASTRA_BuildLight', (0.15, 0.17, 0.19), metallic=0.15, roughness=0.48)
BUILD_GLASS = mat('ASTRA_BuildGlass', (0.015, 0.07, 0.11), metallic=0.48, roughness=0.24)
UAV_WHITE = mat('ASTRA_UAVWhite', (0.74, 0.77, 0.80), metallic=0.25, roughness=0.30)
UAV_DARK = mat('ASTRA_UAVDark', (0.014, 0.021, 0.028), metallic=0.74, roughness=0.22)
UAV_ACCENT = mat('ASTRA_UAVAccent', (0.55, 0.055, 0.018), metallic=0.30, roughness=0.25)
BEACON_MAT = mat('ASTRA_Beacon', (0.22, 0.004, 0.002), roughness=0.15,
                 emission=(1.0, 0.0, 0.0), emission_strength=25.0)
LASER_MAT = mat('ASTRA_Laser', (0.65, 0.002, 0.0), roughness=0.12,
                emission=(1.0, 0.02, 0.0), emission_strength=60.0)
GLOW_MAT = mat('ASTRA_LaserGlow', (0.65, 0.002, 0.0), roughness=0.12,
               emission=(1.0, 0.0, 0.0), emission_strength=8.0)
try:
    GLOW_MAT.surface_render_method = 'BLENDED'
    GLOW_MAT.node_tree.nodes.get('Principled BSDF').inputs['Alpha'].default_value = 0.30
except Exception:
    pass
ACK_MAT = mat('ASTRA_Ack', (0.01, 0.4, 0.05), roughness=0.2,
              emission=(0.1, 1.0, 0.2), emission_strength=25.0)
HUD_MAT = mat('ASTRA_HUD', (0.035, 0.08, 0.10), roughness=0.45,
              emission=(0.22, 0.70, 0.95), emission_strength=3.5)
HUD_GOOD = mat('ASTRA_HUD_GOOD', (0.015, 0.22, 0.06), roughness=0.40,
               emission=(0.15, 1.0, 0.20), emission_strength=4.0)
HUD_WARN = mat('ASTRA_HUD_WARN', (0.32, 0.11, 0.005), roughness=0.40,
               emission=(1.0, 0.28, 0.01), emission_strength=4.0)
CLOUD = mat('ASTRA_Cloud', (0.30, 0.33, 0.36), roughness=1.0)
STAR = mat('ASTRA_Star', (1, 1, 1), emission=(1.0, 1.0, 1.0), emission_strength=8.0)
STYLES = {'norm': HUD_MAT, 'good': HUD_GOOD, 'warn': HUD_WARN}

# ----------------------------------------------------------------------------
# World, sun, city, stars
# ----------------------------------------------------------------------------
world = bpy.data.worlds.get('ASTRA_WORLD') or bpy.data.worlds.new('ASTRA_WORLD')
scene.world = world
try:
    world.use_nodes = True
except Exception:
    pass
wn = world.node_tree.nodes
wl = world.node_tree.links
wn.clear()
wo = wn.new('ShaderNodeOutputWorld')
wb = wn.new('ShaderNodeBackground')
wb.inputs['Color'].default_value = (0.006, 0.014, 0.026, 1.0)
wb.inputs['Strength'].default_value = 0.38
wl.new(wb.outputs['Background'], wo.inputs['Surface'])

sun_data = bpy.data.lights.new('ASTRA_SUN_DATA', 'SUN')
sun_data.energy = 2.0
sun_data.angle = math.radians(20)
sun = bpy.data.objects.new('ASTRA_SUN', sun_data)
ASTRA_ENV.objects.link(sun)
sun.rotation_euler = (math.radians(28), math.radians(-24), math.radians(28))

add_cube('ASTRA_GROUND', (0, 0, -2), (150, 150, 2), GROUND, ASTRA_ENV)
for x in (-100, -50, 0, 50, 100):
    add_cube('ASTRA_ROAD_V_%d' % x, (x, 0, 0.02), (4.5, 145, 0.06), ROAD, ASTRA_ENV)
    add_cube('ASTRA_SIDE_VL_%d' % x, (x - 5.2, 0, 0.05), (0.65, 145, 0.05), SIDEWALK, ASTRA_ENV)
    add_cube('ASTRA_SIDE_VR_%d' % x, (x + 5.2, 0, 0.05), (0.65, 145, 0.05), SIDEWALK, ASTRA_ENV)
for y in (-100, -50, 0, 50, 100):
    add_cube('ASTRA_ROAD_H_%d' % y, (0, y, 0.02), (145, 4.5, 0.06), ROAD, ASTRA_ENV)

city_rng = random.Random(9301)
for ix in range(-6, 7):
    for iy in range(-5, 6):
        if (ix + 2 * iy) % 4 == 0:
            continue
        x = ix * 19 + city_rng.uniform(-2.0, 2.0)
        y = iy * 20 + city_rng.uniform(-2.0, 2.0)
        w = city_rng.uniform(4.2, 7.4)
        d = city_rng.uniform(4.2, 7.4)
        h = city_rng.uniform(9, 38)
        if abs(x) < 55 and abs(y) < 55:
            h = city_rng.uniform(10, 26)
        m_ = [BUILD_DARK, BUILD_LIGHT, BUILD_GLASS][city_rng.randrange(3)]
        add_cube('ASTRA_BUILDING_%d_%d' % (ix, iy), (x, y, h / 2), (w, d, h / 2),
                 m_, ASTRA_CITY, bevel=0.12)

cloud_rng = random.Random(7717)
for i in range(10):
    base = Vector((cloud_rng.uniform(-120, 120), cloud_rng.uniform(-100, 120),
                   cloud_rng.uniform(130, 165)))
    for j in range(3):
        p = base + Vector((j * 4.5, cloud_rng.uniform(-2, 2), cloud_rng.uniform(-1.5, 1.5)))
        c = add_uv('ASTRA_CLOUD_%d_%d' % (i, j), tuple(p), cloud_rng.uniform(3.0, 4.7),
                   CLOUD, ASTRA_ENV, segments=8, rings=5)
        c.scale = (1.5, 0.8, 0.65)

star_rng = random.Random(4242)
for i in range(220):
    a = star_rng.uniform(0.0, 2.0 * math.pi)
    zz = star_rng.uniform(0.03, 0.95)
    rr = math.sqrt(1.0 - zz * zz)
    add_uv('ASTRA_STAR_%d' % i, (1600 * rr * math.cos(a), 1600 * rr * math.sin(a), 1600 * zz),
           star_rng.uniform(0.5, 1.1), STAR, ASTRA_ENV, segments=6, rings=4)

# ----------------------------------------------------------------------------
# UAV model
# ----------------------------------------------------------------------------
def make_uav(name, collection, scale=1.0):
    root = add_empty(name, (0, 0, 0), collection, 'CUBE')

    fus = add_cube(name + '_FUSELAGE', (0, 0, 0), (3.6 * scale, 0.62 * scale, 0.38 * scale),
                   UAV_DARK, collection, 0.16 * scale)
    fus.parent = root
    nose = add_uv(name + '_NOSE', (3.05 * scale, 0, 0), 0.75 * scale, UAV_WHITE, collection, 12, 6)
    nose.parent = root

    for side in (-1, 1):
        tag = 'L' if side < 0 else 'R'
        wing = add_cube(name + '_WING_' + tag, (-0.15 * scale, side * 2.45 * scale, 0),
                        (2.1 * scale, 0.32 * scale, 0.12 * scale), UAV_WHITE, collection, 0.06 * scale)
        wing.parent = root
        wing.rotation_euler.z = side * math.radians(7)
        tip = add_cube(name + '_TIP_' + tag, (-0.75 * scale, side * 4.2 * scale, -0.05 * scale),
                       (1.05 * scale, 0.23 * scale, 0.09 * scale), UAV_ACCENT, collection, 0.04 * scale)
        tip.parent = root
        tip.rotation_euler.z = side * math.radians(7)
        fin = add_cube(name + '_FIN_' + tag, (-2.5 * scale, side * 0.78 * scale, 0.56 * scale),
                       (0.85 * scale, 0.10 * scale, 0.65 * scale), UAV_WHITE, collection, 0.04 * scale)
        fin.parent = root
        fin.rotation_euler.y = side * math.radians(16)

    for idx, (rx, ry) in enumerate(((1.0, 2.6), (1.0, -2.6), (-2.0, 2.1), (-2.0, -2.1))):
        mast = add_cyl(name + '_MAST_%d' % idx, (rx * scale, ry * scale, 0.72 * scale),
                       0.055 * scale, 0.64 * scale, UAV_DARK, collection)
        mast.parent = root
        prop = add_cube(name + '_PROP_%d' % idx, (rx * scale, ry * scale, 1.08 * scale),
                        (0.90 * scale, 0.04 * scale, 0.025 * scale), UAV_DARK, collection)
        prop.parent = root
        prop.rotation_euler.z = (idx % 2) * math.pi / 2
    return root


T1 = make_uav('TERMINAL_1', ASTRA_T1, 1.35)
T2 = make_uav('TERMINAL_2', ASTRA_T2, 1.35)

# ---- T1 nose mounted pan / tilt gimbal + camera ---------------------------
mount = add_cube('ASTRA_NOSE_MOUNT', (NOSE_X - 0.30, 0, NOSE_Z), (0.40, 0.42, 0.20), UAV_DARK, ASTRA_T1, 0.06)
mount.parent = T1

GIMBAL_BASE = add_empty('T1_GIMBAL_BASE', (NOSE_X, 0, NOSE_Z), ASTRA_T1, 'CUBE')
GIMBAL_BASE.parent = T1
PAN = add_empty('T1_PAN_AXIS', (0, 0, 0), ASTRA_T1, 'CIRCLE')
PAN.parent = GIMBAL_BASE
pan_disk = add_cyl('ASTRA_PAN_DISK', (0, 0, 0), 0.50, 0.14, UAV_DARK, ASTRA_T1, vertices=24)
pan_disk.parent = PAN

TILT = add_empty('T1_TILT_AXIS', tuple(TILT_OFF), ASTRA_T1, 'CIRCLE')
TILT.parent = PAN
pod = add_cube('ASTRA_CAMERA_POD', (0.70, 0, 0), (0.62, 0.42, 0.34), UAV_DARK, ASTRA_T1, 0.07)
pod.parent = TILT
lens_ring = add_cyl('ASTRA_LENS_RING', (1.36, 0, 0), 0.30, 0.20, UAV_DARK, ASTRA_T1,
                    rotation=(0, math.pi / 2, 0), vertices=24)
lens_ring.parent = TILT
LENS = add_cyl('T1_LENS', (1.47, 0, 0), 0.21, 0.14, LASER_MAT, ASTRA_T1,
               rotation=(0, math.pi / 2, 0), vertices=24)
LENS.parent = TILT

camera_data = bpy.data.cameras.new('T1_TRACKING_CAMERA_DATA')
T1_CAMERA = bpy.data.objects.new('T1_TRACKING_CAMERA', camera_data)
ASTRA_CAM.objects.link(T1_CAMERA)
T1_CAMERA.parent = TILT
T1_CAMERA.location = tuple(CAM_OFF)
T1_CAMERA.rotation_mode = 'XYZ'
camera_data.clip_start = 0.03
camera_data.clip_end = 5000
camera_data.sensor_fit = 'VERTICAL'
camera_data.lens = camera_data.sensor_height / (2.0 * TAN_Y)

# ---- T2 beacon (on a mast above the nose, visible from behind) ------------
mast = add_cyl('T2_BEACON_MAST', (BEACON_LOCAL.x, 0, 1.25), 0.06, 0.50, UAV_DARK, ASTRA_T2)
mast.parent = T2
BEACON = add_uv('T2_BEACON', tuple(BEACON_LOCAL), 0.25, BEACON_MAT, ASTRA_T2, 16, 8)
BEACON.parent = T2
for r_ in (0.45, 0.65):
    ring = add_torus('ASTRA_BEACON_RING_%d' % int(r_ * 100), r_, 0.02, LASER_MAT, ASTRA_FX,
                     rotation=(0, math.pi / 2, 0))
    ring.parent = T2
    ring.location = tuple(BEACON_LOCAL)

# ---- name tags ------------------------------------------------------------
LABELS = []
for root, txt in ((T1, 'T1 | TRACKER'), (T2, 'T2 | BEACON')):
    lab = add_text('ASTRA_LABEL_' + root.name, txt, root, (0, 0, 6.0), 2.4, HUD_MAT, ASTRA_FX, 'CENTER')
    cst = lab.constraints.new('TRACK_TO')
    cst.track_axis = 'TRACK_Z'
    cst.up_axis = 'UP_Y'
    LABELS.append((lab, cst))

# ---- presentation cameras --------------------------------------------------
TPP_LENS, OPEN_LENS = 30.0, 38.0
TPP_DATA = bpy.data.cameras.new('T1_TPP_CAMERA_DATA')
TPP_CAMERA = bpy.data.objects.new('T1_TPP_CAMERA', TPP_DATA)
ASTRA_CAM.objects.link(TPP_CAMERA)
TPP_DATA.lens = TPP_LENS
TPP_DATA.sensor_width = 36
TPP_DATA.clip_end = 5000
OPEN_DATA = bpy.data.cameras.new('OPEN_OBSERVER_CAMERA_DATA')
OPEN_CAMERA = bpy.data.objects.new('OPEN_OBSERVER_CAMERA', OPEN_DATA)
ASTRA_CAM.objects.link(OPEN_CAMERA)
OPEN_DATA.lens = OPEN_LENS
OPEN_DATA.sensor_width = 36
OPEN_DATA.clip_end = 5000
CAMS = {'tpp': TPP_CAMERA, 'onboard': T1_CAMERA, 'open': OPEN_CAMERA}

# ---- laser beam + optical / ack packets ------------------------------------
def make_beam(name, radius, material):
    d = bpy.data.curves.new(name + '_DATA', 'CURVE')
    d.dimensions = '3D'
    d.bevel_depth = radius
    d.bevel_resolution = 2
    d.materials.append(material)
    o = bpy.data.objects.new(name, d)
    ASTRA_FX.objects.link(o)
    sp = d.splines.new('POLY')
    sp.points.add(1)
    o.hide_viewport = True
    o.hide_render = True
    return o, sp


BEAM_CORE, SPL_CORE = make_beam('FSOC_BEAM', 0.09, LASER_MAT)
BEAM_GLOW, SPL_GLOW = make_beam('FSOC_BEAM_GLOW', 0.26, GLOW_MAT)
PULSES = []
for i in range(12):
    p = add_uv('ASTRA_OPTICAL_PACKET_%d' % i, (0, 0, 0), 0.34, LASER_MAT, ASTRA_FX, 8, 5)
    p.hide_viewport = p.hide_render = True
    PULSES.append(p)
ACKS = []
for i in range(6):
    p = add_uv('ASTRA_ACK_PACKET_%d' % i, (0, 0, 0), 0.22, ACK_MAT, ASTRA_FX, 8, 5)
    p.hide_viewport = p.hide_render = True
    ACKS.append(p)

# ----------------------------------------------------------------------------
# HUD (3D text parented to each camera; only the active set is visible)
# ----------------------------------------------------------------------------
def build_hud(cam, prefix, hw, hh, depth, n, size, left, top, banner_y, banner_size):
    lines = []
    pitch = size * 1.55
    for i in range(n):
        lines.append(add_text('%s%d' % (prefix, i), '-', cam, (left, top - i * pitch, -depth),
                              size, HUD_MAT, ASTRA_FX))
    banner = add_text(prefix + 'BANNER', '-', cam, (0.0, banner_y, -depth), banner_size,
                      HUD_MAT, ASTRA_FX, 'CENTER')
    return {'lines': lines, 'banner': banner, 'objs': lines + [banner]}


HUD = {}
OB_DEPTH = 10.0
ob_hh = OB_DEPTH * TAN_Y
ob_hw = ob_hh * LOGICAL_W / LOGICAL_H
HUD['onboard'] = build_hud(T1_CAMERA, 'ASTRA_HUD_ONBOARD_', ob_hw, ob_hh, OB_DEPTH, 9,
                           ob_hh * 0.050, -ob_hw * 0.94, ob_hh * 0.88, -ob_hh * 0.86, ob_hh * 0.052)

# sensor window frame + crosshair (the 640x480 virtual sensor)
def hud_box(name, loc, scale):
    o = add_cube(name, (0, 0, 0), scale, HUD_MAT, ASTRA_FX)
    o.parent = T1_CAMERA
    o.location = loc
    HUD['onboard']['objs'].append(o)
    return o


th = ob_hh * 0.006
hud_box('ASTRA_SENSOR_TOP', (0, ob_hh - th, -OB_DEPTH), (ob_hw, th, 0.002))
hud_box('ASTRA_SENSOR_BOT', (0, -ob_hh + th, -OB_DEPTH), (ob_hw, th, 0.002))
hud_box('ASTRA_SENSOR_L', (-ob_hw + th, 0, -OB_DEPTH), (th, ob_hh, 0.002))
hud_box('ASTRA_SENSOR_R', (ob_hw - th, 0, -OB_DEPTH), (th, ob_hh, 0.002))
gap, ln = ob_hh * 0.03, ob_hh * 0.05
hud_box('ASTRA_XH_R', (gap + ln, 0, -OB_DEPTH), (ln, th * 0.7, 0.002))
hud_box('ASTRA_XH_L', (-gap - ln, 0, -OB_DEPTH), (ln, th * 0.7, 0.002))
hud_box('ASTRA_XH_U', (0, gap + ln, -OB_DEPTH), (th * 0.7, ln, 0.002))
hud_box('ASTRA_XH_D', (0, -gap - ln, -OB_DEPTH), (th * 0.7, ln, 0.002))

RETICLE = add_torus('ASTRA_DETECTION_RETICLE', ob_hh * 0.14, ob_hh * 0.006, HUD_GOOD, ASTRA_FX)
RETICLE.parent = T1_CAMERA
PRED_MARK = add_torus('ASTRA_PREDICTION_MARK', ob_hh * 0.05, ob_hh * 0.004, HUD_MAT, ASTRA_FX, segs=20)
PRED_MARK.parent = T1_CAMERA
HUD['onboard']['objs'] += [RETICLE, PRED_MARK]

EXT_DEPTH = 5.0
for key, cam, lens in (('tpp', TPP_CAMERA, TPP_LENS), ('open', OPEN_CAMERA, OPEN_LENS)):
    hw_ = EXT_DEPTH * 18.0 / lens
    hh_ = hw_ * 9.0 / 16.0
    HUD[key] = build_hud(cam, 'ASTRA_HUD_%s_' % key.upper(), hw_, hh_, EXT_DEPTH, 13,
                         hh_ * 0.036, -hw_ * 0.95, hh_ * 0.90, -hh_ * 0.90, hh_ * 0.046)

VIEW = {'active': None}
BODY_CACHE = {}
STYLE_CACHE = {}


def set_text(o, txt, style):
    if BODY_CACHE.get(o.name) != txt:
        o.data.body = txt
        BODY_CACHE[o.name] = txt
    if STYLE_CACHE.get(o.name) != style:
        o.data.materials[0] = STYLES[style]
        STYLE_CACHE[o.name] = style


def set_view(which):
    VIEW['active'] = which
    scene.camera = CAMS[which]
    for k, hud in HUD.items():
        show = (k == which)
        for o in hud['objs']:
            o.hide_viewport = not show
            o.hide_render = not show
    for lab, cst in LABELS:
        cst.target = CAMS[which]
        lab.hide_viewport = lab.hide_render = (which == 'onboard')


def update_reticle(active):
    S = SIM
    m = S['measurement']
    show = active and m is not None
    RETICLE.hide_viewport = RETICLE.hide_render = not show
    if show:
        RETICLE.location = ((m.x - 0.5) * K_X * OB_DEPTH, (m.y - 0.5) * K_Y * OB_DEPTH, -OB_DEPTH)
        set_text_style_mat(RETICLE, 'good' if S['state'] == 'LOCKED' else 'warn')
    pr = S['prediction']
    showp = active and pr is not None
    PRED_MARK.hide_viewport = PRED_MARK.hide_render = not showp
    if showp:
        PRED_MARK.location = ((pr.x - 0.5) * K_X * OB_DEPTH, (pr.y - 0.5) * K_Y * OB_DEPTH, -OB_DEPTH)


def set_text_style_mat(o, style):
    if STYLE_CACHE.get(o.name) != style:
        o.data.materials[0] = STYLES[style]
        STYLE_CACHE[o.name] = style


def update_laser(t):
    S = SIM
    active = S['link_established'] and t >= LINK_START
    for o in (BEAM_CORE, BEAM_GLOW):
        o.hide_viewport = o.hide_render = not active
    txing = active and t >= TRANSFER_START and not S['transfer_complete']
    for o in PULSES + ACKS:
        o.hide_viewport = o.hide_render = not txing
    if not active:
        return
    a, b = S['lens'], S['beacon']
    for sp in (SPL_CORE, SPL_GLOW):
        sp.points[0].co = (a.x, a.y, a.z, 1.0)
        sp.points[1].co = (b.x, b.y, b.z, 1.0)
    if txing:
        dist = max((b - a).length, 1.0)
        for i, o in enumerate(PULSES):
            o.location = a.lerp(b, ((t - LINK_START) * 240.0 / dist + i / 12.0) % 1.0)
        for i, o in enumerate(ACKS):
            o.location = b.lerp(a, ((t - LINK_START) * 160.0 / dist + i / 6.0) % 1.0)


def apply_scene(frame):
    S = SIM
    t = S['t']
    T1.location = S['p1']
    T1.rotation_euler = S['e1']
    T2.location = S['p2']
    T2.rotation_euler = S['e2']
    PAN.rotation_euler.z = S['pan_view']
    TILT.rotation_euler.y = S['tilt_view']
    T1_CAMERA.rotation_euler = S['cam_q'].to_euler()

    mid = (S['p1'] + S['p2']) * 0.5
    aim(TPP_CAMERA, mid + Vector((-25.0 + 2.5 * t, -95.0, 22.0 - 0.3 * t)), mid + Vector((0, 0, -1)))
    th_ = math.radians(-25.0 + 4.0 * (t - CUT_EXTERNAL))
    aim(OPEN_CAMERA, mid + Vector((118.0 * math.sin(th_), -118.0 * math.cos(th_), 32.0)), mid)

    which = 'tpp' if t < SEARCH_START else ('onboard' if t < CUT_EXTERNAL else 'open')
    if which != VIEW['active']:
        set_view(which)

    ext, ob, (banner, bstyle) = hud_content()
    hud = HUD[which]
    rows = ob if which == 'onboard' else ext
    for o, (txt, style) in zip(hud['lines'], rows):
        set_text(o, txt, style)
    set_text(hud['banner'], banner, bstyle)
    update_reticle(which == 'onboard')
    update_laser(t)


def astra_link_handler(scene_arg, depsgraph=None):
    frame = int(clamp(scene_arg.frame_current, 1, END_FRAME))
    if SIM['frame'] is None or frame < SIM['frame']:
        reset_sim()
        start = 1
    else:
        start = SIM['frame'] + 1
    for f in range(start, frame + 1):
        sim_step(f)
    apply_scene(frame)


# ----------------------------------------------------------------------------
# Timeline markers, compositor glow, metadata
# ----------------------------------------------------------------------------
for mk in list(scene.timeline_markers):
    scene.timeline_markers.remove(mk)
for label, sec in (('PATROL', 0.0), ('COARSE_SEARCH', SEARCH_START), ('MANEUVER_CT', MANEUVER_START),
                   ('DROPOUT_REACQ', DROPOUT[0]), ('VERIFY', VERIFY_START),
                   ('FSOC_LINK', LINK_START), ('TRANSFER', TRANSFER_START), ('COMPLETE', 29.0)):
    scene.timeline_markers.new(label, frame=int(round(sec * FPS)) + 1)

try:
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new('CompositorNodeRLayers')
    glare = nt.nodes.new('CompositorNodeGlare')
    glare.glare_type = 'FOG_GLOW'
    glare.quality = 'MEDIUM'
    glare.threshold = 0.8
    glare.size = 6
    comp = nt.nodes.new('CompositorNodeComposite')
    nt.links.new(rl.outputs['Image'], glare.inputs['Image'])
    nt.links.new(glare.outputs['Image'], comp.inputs['Image'])
except Exception:
    pass  # emissive beam + thick glow tube still render without compositor glare

scene['ASTRA_VERSION'] = 'FINAL_JURY_30S_V7'
scene['ASTRA_ALGORITHMS'] = 'SAPTL | Kalman prediction | Adaptive CV/CT | PID + Velocity Feedforward'
scene['ASTRA_LOGICAL_CAMERA'] = '640 x 480, ~12 x 9 deg'
scene['ASTRA_TRANSFER'] = '2.0 GB demonstration payload, Selective Repeat ARQ, 1.80 Gbps nominal demo rate'

# ----------------------------------------------------------------------------
# Go
# ----------------------------------------------------------------------------
reset_sim()
set_view('tpp')
scene.frame_set(1)
bpy.app.handlers.frame_change_pre.append(astra_link_handler)
astra_link_handler(scene)

print('=' * 78)
print('ASTRA LINK V7 READY  |  0-10 patrol | 10-12 search+lock | 13-19 CV/CT maneuver')
print('16-17 dropout+reacq | 17.6-18.5 verify | 18.9-30 laser + ARQ transfer')
print('Press SPACE from frame 1. Render only after previewing.')
print('=' * 78)