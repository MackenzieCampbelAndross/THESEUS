# AstraLink - AI-based Virtual Camera Tracking for FSOC Coarse Alignment
Team Theseus | Smart India Hackathon 2026 | PS SIH26169 | Software / Smart Automation

A hardware-independent simulation of coarse alignment for a mobile Free-Space Optical Communication
terminal. Nothing is scripted to "succeed": every metric is computed from a closed loop that only sees
rendered 640x480 camera frames and gimbal encoder readings. Ground truth is used **only** for scoring.

## Run
    pip install -r requirements.txt
    python main.py                                        # 60 s, mixed motion, haze, secure link
    python main.py --scenario aggressive --weather fog --video
    python main.py --runs 20 --no-comms                   # Monte-Carlo over seeds
    python main.py --show                                 # live window (ESC to stop)
Outputs go to `results/`: dashboard PNG, per-frame CSV log, summary JSON, optional annotated MP4.
Options: `--scenario straight|circular|mixed|aggressive`, `--weather clear|haze|fog|dense_fog`,
`--cue-sigma DEG` (error of the a-priori pointing cue), `--seed`, `--duration`.

## Architecture (matches the technical-approach slide)
| Module | Role |
|---|---|
| `world.py` | Virtual scene: 3-D moving terminal, 640x480 camera, Beer-Lambert fog attenuation, scintillation, beam wander, motion blur, platform sway/vibration, sensor + hot-pixel noise, random occlusions, world-fixed clutter lights, pan/tilt gimbal (rate/accel limits, latency) |
| `detector.py` | Background suppression, MAD-based adaptive threshold, blob centroid, SNR-based confidence, dark-frame fixed-pattern calibration |
| `tracker.py` | Two Kalman filters (CV, and CT with online turn-rate estimate); model chosen from accumulated innovation likelihood with hysteresis (SAPTL / adaptive CV-CT switching) |
| `alignment.py` | LOCKED / COASTING / LOST state machine, confidence-gated data association, clutter map, prediction-guided Levy-flight search (Mantegna), gimbal control with feed-forward |
| `comms.py` | TLS 1.3 (mutual certificate authentication, real OpenSSL) -> packetization (CRC32) -> lossy optical channel whose loss follows the *actual* pointing error -> Selective-Repeat ARQ (cumulative ACK + SACK bitmap) |
| `runner.py` | Closed-loop harness, scoring against the target outcomes |

## Design notes / honest assumptions
* ARQ sits *below* TLS so TLS sees a reliable ordered stream (TLS requires that); the handshake itself crosses the lossy link.
* The tracker starts from a coarse a-priori cue (GPS/ephemeris-style) - never from the true position.
* Static background lights are handled with a learned clutter map and a moving-or-dominant check before a lock is accepted;
  a beacon that is stationary AND dim in a scene with similar-brightness static lights can be missed. A real system
  would use beacon modulation/wavelength filtering.
* Reacquisition time is counted from when the beacon is physically observable again (not from the start of an occlusion).
* Acquisition is scored against ground truth (a lock on the wrong light does not count).
* Not implemented: ONNX Runtime (no learned model is used; the "adaptive" part is the online CV/CT model selection) and a GUI beyond the OpenCV preview.

## Reference results (this code, seed 1, mixed/haze, 60 s, with secure link)
Acquisition 0.17 s, tracking RMS 4.2 px, lock-loss 0 %, reacquisition 0 s (no long losses), ~160 FPS,
TLS 1.3 handshake OK, 14.6 MB delivered with matching SHA-256 and 5 % retransmissions.
Monte-Carlo (10 seeds, mixed/haze, 40 s): median acquisition 0.17 s, median tracking RMS 4.5 px, but one seed failed to
acquire because the initial cue put the beacon far outside the field of view. Failures are reported, not hidden.
