"""All tunable parameters live here. Nothing in the pipeline is hard-wired to a result."""
from dataclasses import dataclass, field, asdict


@dataclass
class CameraCfg:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    hfov_deg: float = 24.0          # wide-FOV acquisition camera
    exposure_frac: float = 0.5      # fraction of frame time the shutter is open (motion blur)
    read_noise: float = 3.0         # grey levels
    shot_noise_k: float = 0.45
    n_hot_pixels: int = 25
    spot_sigma_px: float = 1.6      # in-focus beacon spot sigma
    beacon_ref_amp: float = 900.0   # peak grey level at 100 m in perfect visibility
    beacon_ref_range_m: float = 100.0


@dataclass
class GimbalCfg:
    v_max: float = 90.0             # deg/s
    a_max: float = 500.0            # deg/s^2
    kv: float = 30.0                # inner velocity-loop bandwidth (1/s)
    encoder_noise_deg: float = 0.004
    el_min: float = -20.0
    el_max: float = 85.0


@dataclass
class ScenarioCfg:
    kind: str = "mixed"             # straight | circular | mixed | aggressive
    weather: str = "haze"           # clear | haze | fog | dense_fog
    duration_s: float = 60.0
    speed_mps: float = 18.0
    start_range_m: tuple = (220.0, 380.0)
    start_alt_m: tuple = (30.0, 120.0)
    min_range_m: float = 70.0
    max_range_m: float = 450.0
    cue_sigma_deg: float = 10.0     # error of the coarse a-priori pointing cue (e.g. GPS/ephemeris)
    sway_deg: float = 1.2           # slow platform sway of Terminal 1 (unknown to tracker)
    occlusion_rate_per_min: float = 1.5
    occlusion_dur_s: tuple = (0.3, 1.2)


WEATHER = {  # visibility_km, veil grey level, scintillation sigma, beam wander px, vibration px, distractors
    "clear":     dict(vis_km=20.0, veil=18.0, scint=0.10, wander=0.4, vib=1.0, n_distract=4),
    "haze":      dict(vis_km=4.0,  veil=40.0, scint=0.22, wander=0.8, vib=1.6, n_distract=6),
    "fog":       dict(vis_km=1.2,  veil=70.0, scint=0.30, wander=1.0, vib=2.0, n_distract=6),
    "dense_fog": dict(vis_km=0.7,  veil=95.0, scint=0.40, wander=1.4, vib=2.5, n_distract=8),
}


@dataclass
class DetectorCfg:
    blur_sigma: float = 1.5
    bg_kernel: int = 41
    k_sigma: float = 5.0
    min_area: int = 3
    max_area: int = 600
    max_aspect: float = 3.5
    snr0: float = 7.0
    snr_scale: float = 2.5
    max_candidates: int = 8


@dataclass
class TrackerCfg:
    q_cv: float = 25.0              # white-accel std (deg/s^2) for the CV model
    q_ct: float = 8.0               # for the CT model (smaller: curvature is modelled)
    sigma_px: float = 1.2           # measurement sigma at confidence 1
    conf_inflate: float = 4.0       # R inflation for low-confidence measurements
    omega_alpha: float = 0.12       # EMA factor for the turn-rate estimate
    omega_min: float = 0.6          # deg/s heading rate needed to consider CT
    ll_forget: float = 0.93         # forgetting factor of the model likelihood
    switch_on: float = 1.5          # log-likelihood margin CV->CT
    switch_off: float = 0.5         # margin CT->CV
    coast_damp: float = 0.985       # velocity damping per frame while unobserved


@dataclass
class ControlCfg:
    c_track: float = 0.35           # min confidence to update an existing track
    c_acquire: float = 0.6          # min confidence to start a track
    confirm_frames: int = 6
    move_deg: float = 0.45          # world-angle displacement that proves a candidate is moving
    lone_snr: float = 30.0          # SNR needed to accept a lone, non-moving candidate
    dominance: float = 2.0          # SNR ratio vs. other blobs that marks a clear brightness outlier
    confirm_gate_deg: float = 1.2
    coast_frames: int = 15          # ~0.5 s of prediction-only before declaring LOST
    gate_min_deg: float = 0.9
    gate_max_deg: float = 2.5
    kp: float = 9.0                 # position gain (1/s)
    lead_frames: float = 1.5
    search_v_max: float = 18.0      # deg/s: keeps motion blur small enough to detect
    search_kp: float = 6.0
    levy_beta: float = 1.5
    levy_scale_deg: float = 6.0
    levy_r0_deg: float = 4.0
    levy_r_growth_deg_s: float = 8.0
    levy_r_max_deg: float = 20.0
    levy_dwell_s: float = 0.4


@dataclass
class CommsCfg:
    enabled: bool = True
    mtu: int = 1000
    window: int = 32
    micro_dt_s: float = 0.002       # link time-step
    frames_per_micro: int = 1       # => 500 kB/s raw
    prop_delay_steps: int = 1
    rto_steps: int = 14
    base_loss: float = 0.02         # turbulence fading floor on the optical channel
    bit_error_prob: float = 0.01    # per-frame corruption probability (caught by CRC32)
    coupling_sigma_px: float = 30.0 # pointing error at which coupling drops to 1/e
    app_bytes_per_frame: int = 8192
    app_backlog_limit: int = 65536


@dataclass
class Targets:
    acquisition_s: float = 2.0
    tracking_err_px: float = 10.0
    lock_loss_rate: float = 0.05
    reacq_s: float = 1.0
    fps: float = 20.0


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    gimbal: GimbalCfg = field(default_factory=GimbalCfg)
    scenario: ScenarioCfg = field(default_factory=ScenarioCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    comms: CommsCfg = field(default_factory=CommsCfg)
    targets: Targets = field(default_factory=Targets)

    def to_dict(self):
        return asdict(self)
