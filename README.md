# AstraLink

**Autonomous AI-Based Virtual Camera Tracking and Secure Communication for Mobile Free-Space Optical Communication (FSOC) Coarse Alignment**

Team Theseus | Smart India Hackathon 2026 | Problem Statement: PS SIH26169 | Category: Software / Smart Automation

---

## Executive Summary

Free-Space Optical Communication (FSOC) delivers multi-gigabit data transmission rates with low probability of intercept and immunity to radio-frequency (RF) jamming. However, optical communication requires narrow divergence beams (typically sub-milliradian to a few milliradians), making coarse alignment and continuous beam tracking between mobile aerial terminals (such as UAVs and tactical aircraft) an acute engineering challenge. Atmospheric turbulence, fog/haze extinction, platform sway, high-g maneuvers, and sensor noise cause optical link disruptions.

AstraLink is a hardware-independent, closed-loop simulation and verification system designed to solve the FSOC coarse-alignment and secure transmission problem. The platform integrates:

1. A deterministic 3D flight dynamics and electro-optical simulation engine with forward-looking infrared (FLIR) HUD visualization.
2. An optical tracking system utilizing dual-model Kalman filters (Constant Velocity and Constant Turn with adaptive likelihood switching), motion consistency verification, and prediction-guided reacquisition search (Lévy flight and raster sweep).
3. A multi-layer cryptographic identity, zero-trust evaluation gate, AES-256-GCM authenticated encryption pipeline, and Selective Repeat ARQ flow control with atomic checkpoint persistence.
4. An end-to-end telemetry bridge connecting the browser-based 3D simulation to the Python tracking and security subsystems in real time.

---

## System Architecture

The project is structured into three synchronized subsystems linked via an integration bridge:

```
+-------------------------------------------------------------------------------+
|                       MEMBER 1: 3D SIMULATION ENGINE                          |
|                       (TypeScript / Three.js / WebGL)                         |
|  - Tactical UAV flight kinematics with coordinated banking physics            |
|  - Transceiver turret electro-optical camera POV & FLIR targeting HUD         |
|  - Physical Rayleigh/Mie atmospheric scattering & terrain rendering           |
|  - Deterministic fixed-dt clock (1/30 s) with Seeded PRNG                     |
+-------------------------------------------------------------------------------+
                                      |
                                      | WebSocket (ws://127.0.0.1:8765)
                                      v
+-------------------------------------------------------------------------------+
|                         INTEGRATION TELEMETRY BRIDGE                          |
|                    (Python / asyncio / websockets / adapter)                  |
|  - Bidirectional IPC linking 3D renderer to Python tracking and security      |
|  - Timestamp normalization, clock rewind handling, and state adaptation       |
+-------------------------------------------------------------------------------+
                                      |
                   +------------------+------------------+
                   |                                     |
                   v                                     v
+------------------------------------+ +----------------------------------------+
|      MEMBER 2: OPTICAL TRACKING    | |      MEMBER 3: SECURITY & TRANSPORT    |
|              (Python)              | |                 (Python)               |
| - Virtual camera projection        | | - Mutual authentication (Ed25519)      |
| - Adaptive Kalman filter (CV / CT) | | - 5-Gate trust authorization engine    |
| - Motion consistency verification  | | - Ephemeral ECDH key exchange (X25519) |
| - PID + velocity feedforward PTZ   | | - AES-256-GCM authenticated encryption |
| - Lévy flight / raster reacq search| | - Selective Repeat ARQ sliding window  |
| - 336 unit & regression tests      | | - Atomic checkpoint & link recovery    |
+------------------------------------+ +----------------------------------------+
```

---

## Repository Structure

```
THESIUS/
├── .github/
│   └── workflows/
│       └── ci.yml               Automated CI pipeline (Node.js + Python test suites)
├── astralink/                   Self-contained 1-click Python FSOC simulation
│   ├── astralink/               Simulation modules (world, detector, tracker, alignment, comms)
│   ├── main.py                  Benchmark CLI runner and dashboard generator
│   └── requirements.txt         Standalone simulator dependencies
├── astra-link-member2/          Member 2: Tracking, state estimation and control
│   ├── src/                     Core tracking, Kalman, PID, and reacquisition implementations
│   ├── tests/                   Unit and regression test suite (336 tests)
│   └── conftest.py              Isolated pytest path configuration
├── astra_link_member3/          Member 3: Security, authentication and reliable transport
│   ├── src/                     Identity, authentication, trust engine, session, transport
│   ├── tests/                   Cryptographic and attack test suite (203 tests)
│   ├── outputs/reports/         Pre-computed security and transport verification reports
│   └── conftest.py              Isolated pytest path configuration
├── astra_link_integration/      Integration bridge between simulation, tracking, and security
│   ├── adapter.py               State adapter mapping M2 tracking to M3 security inputs
│   ├── bridge_server.py         Real-time WebSocket server (ws://127.0.0.1:8765)
│   ├── pipeline.py              Unified tracking-to-authorization pipeline
│   ├── secure_communication_manager.py End-to-end secure channel coordinator
│   └── tests/                   Bridge and pipeline regression test suite (20 tests)
├── blender/                     3D CAD assets and photorealistic offline animation
│   ├── astra_link_jury_demo.blend Master Blender 5.x scene with embedded simulation script
│   ├── astra_link_level1.py     Extracted 30-second simulation and frame-change handler
│   └── renders/                 Rendered MP4 video, animated GIF, and 8 key still frames
├── docs/                        Official hackathon pitch deck and architecture slides
│   └── SIH2026-THESIUS.pdf      Team Theseus presentation document
├── results/                     Verification evidence, logs, and telemetry
│   ├── telemetry/               Full-run CSV metrics logs (3,900+ frames)
│   ├── security/                Phase 8 adversarial attack verification reports
│   └── dashboards/              Empirical performance and tracking error plots
├── src/                         Member 1: Three.js simulation engine & HUD coordinator
├── tests/                       Member 1: Vitest test suite (32 tests)
├── run_tests.py                 Unified test runner executing all 556 Python tests
├── requirements.txt             Consolidated Python dependencies
├── package.json                 Node.js dependencies and build scripts
├── index.html                   Interactive 3D simulation cockpit and HUD entry point
└── vite.config.ts               Vite bundler configuration
```

---

## Evaluation Guide for the Jury

The codebase provides four evaluation pathways depending on available environment and time:

### Option A: Live 3D Tactical Simulation with Real-Time Python Gateway

Runs the full end-to-end integrated stack: the browser executes the Three.js 3D physics and visualizer, streaming camera detections via WebSocket to the Python bridge, which executes Member 2 Kalman tracking and Member 3 security validation.

1. **Terminal 1: Start the Python Telemetry Bridge**
   ```bash
   pip install -r requirements.txt
   python -m astra_link_integration.bridge_server
   ```
   *Expected output: `[BridgeServer] INFO - Starting Astra Link Telemetry Bridge Server on ws://127.0.0.1:8765...`*

2. **Terminal 2: Launch the 3D Cockpit Interface**
   ```bash
   npm install
   npm run dev
   ```
   Open `http://localhost:5173/` in your browser.

3. **Observed System Behaviors**:
   * **HUD Backend Status**: Indicates `ONLINE (WS 8765)` upon WebSocket connection.
   * **Airspace Search**: The optical gimbal turret performs an autonomous 5 to 10 second sector search pattern across the 3D airspace.
   * **Target Lock**: Upon detecting the target optical beacon, the closed-loop servo slews the gimbal, locks the beacon within the crosshairs, and transitions HUD tracking status to `BEACON LOCKED`.
   * **Security Gateway**: Telemetry indicates `SECURE` once mutual authentication and motion consistency checks pass, authorizing optical laser transmission.
   * **Controls**:
     * `Space`: Pause / Resume simulation.
     * `B`: Break lock (forces gimbal into new sector search).
     * `R`: Reset simulation clock to $t = 0$.
     * `1x / 2x / 4x / 8x`: Optical zoom.
     * `Camera POV / Chase / Tactical`: Perspective switching.

---

### Option B: Single-Command Python Physical Benchmark (5 Seconds)

For automated verification without launching a browser or multiple terminals:

```bash
pip install -r requirements.txt
python astralink/main.py
```

This runs a 60-second closed-loop simulation under atmospheric haze, processing 640x480 virtual camera frames, executing background suppression, blob centroid extraction, dual-mode Kalman filtering, and TLS 1.3 handshake with Selective Repeat ARQ over a pointing-error-driven channel.

**Additional benchmark variations**:
```bash
# Aggressive evasive maneuvers under dense fog with annotated MP4 video output
python astralink/main.py --scenario aggressive --weather fog --video

# 10-seed Monte-Carlo statistical analysis
python astralink/main.py --runs 10 --scenario mixed --no-comms

# Interactive OpenCV tracking preview window
python astralink/main.py --show
```
All outputs (performance dashboards, per-frame CSV logs, and JSON summaries) are saved to `astralink/results/`.

---

### Option C: Complete Automated Test Suite (588 Tests)

The entire codebase is backed by automated test suites validating flight kinematics, tracking algorithms, cryptographic invariants, and adversarial attack resistance:

1. **Run All Python Subsystem Tests (556 Tests)**:
   ```bash
   python run_tests.py
   ```
   *Execution breakdown*:
   * Member 2 (Tracking, Kalman, PID, Reacquisition): 336 passed
   * Member 3 (Cryptography, Mutual Auth, ARQ, Attacks): 203 passed
   * Integration Bridge (Adapter, WebSocket Server, Pipeline): 17 passed

2. **Run Frontend Engine Tests (32 Tests)**:
   ```bash
   npm test
   ```
   *Validates deterministic fixed-step clock, Mulberry32 PRNG reproducibility, flight kinematics, and beacon geometry.*

---

### Option D: Blender 3D CAD Simulation and Video Renderings

The repository includes a 30-second Blender simulation modeling tactical aircraft patrol, optical gimbal slew, corkscrew evasive maneuvers, atmospheric dropout reacquisition, TLS 1.3 verification, and high-bandwidth optical laser transfer.

* **Pre-Rendered Outputs**:
  * Complete 30-second animation: [`blender/renders/astra_link_jury_demo.mp4`](blender/renders/astra_link_jury_demo.mp4) (H.264, 1280x720 at 30 FPS).
  * Animated preview: [`blender/renders/astra_link_jury_demo.gif`](blender/renders/astra_link_jury_demo.gif).
  * Key timeline stills: [`blender/renders/`](blender/renders/) (01 Patrol, 02 Search/Lock, 03 Maneuver, 04 Dropout/Reacq, 05 Verification, 06 Laser Link, 07 ARQ Transfer, 08 Complete).
* **Inspecting or Re-Rendering the Scene**:
  ```bash
  # Open in Blender GUI (Blender 4.2+ or 5.x)
  blender blender/astra_link_jury_demo.blend

  # Headless re-render of key stills
  blender -b blender/astra_link_jury_demo.blend --python blender/astra_link_level1.py
  ```

---

## Empirical Benchmark Results

### 1. Optical Tracking and Alignment Performance

Simulated across 60 seconds with 640x480 resolution virtual camera under atmospheric haze and random occlusions:

| Performance Metric | Target Specification | AstraLink Achieved | Status | Reference Note |
|---|---|---|---|---|
| Initial Acquisition Time | $\le 2.0\text{ s}$ | $0.17\text{ s}$ | PASS | Autonomous search to crosshair lock |
| Tracking Accuracy (RMS) | $\le 10.0\text{ px}$ | $4.15\text{ px}$ | PASS | 95th percentile: $6.84\text{ px}$ |
| Lock-Loss Rate | $< 5.0\%$ | $0.00\%$ | PASS | Maintained through aircraft maneuvers |
| Reacquisition Time (p95) | $\le 1.0\text{ s}$ | $< 0.45\text{ s}$ | PASS | Measured upon beacon re-emergence |
| Loop Processing Rate | $\ge 20\text{ FPS}$ | $145\text{--}205\text{ FPS}$ | PASS | Hardware-independent CPU execution |
| Model Selection Rate | Dynamic | $95.9\%\text{ CT}$ | PASS | Switches CV $\to$ CT on maneuver detection |

### 2. Secure Optical Transport and Communications

Tested over lossy optical channel whose packet loss directly follows real pointing error:

| Parameter | Measured Result | Significance |
|---|---|---|
| Cryptographic Protocol | TLS 1.3 (RFC 8446) / OpenSSL | Real mutual certificate authentication |
| Handshake Completion | $0.37\text{ s}$ | Established immediately post optical lock |
| Transmitted Application Payload | $14.65\text{ MB}$ delivered | 100% SHA-256 integrity match |
| Effective Goodput | $244.1\text{ kB/s}$ | Sustained across turbulent channel |
| Retransmission Ratio | $5.35\%$ | Selective Repeat ARQ recovery |
| Channel Extinction Discards | $144\text{ packets}$ | Corrupted packets dropped via CRC-32 |

### 3. Adversarial Security Verification (Member 3 Threat Model)

The security subsystem enforces five mathematical trust gates before opening the optical transmission channel. Under intentional attack simulation:

| Attack Vector | Simulated Threat Scenario | System Response | Rejection Rate |
|---|---|---|---|
| Cloned Optical Beacon | Attacker mimics target beacon visually without valid Ed25519 key | Mutual authentication fails; state set to `UNTRUSTED`; transmission blocked | 100% |
| Replay Attack | Attacker replays valid authentication tokens across sessions/epochs | Nonce and 5-field cache reject duplicated messages | 100% |
| Packet Tampering | Bit flips injected into ciphertext, nonce, or metadata headers | AES-256-GCM authentication tag verification fails; packet discarded | 100% |
| Forged ACKs | Attacker injects fake ACKs to advance window or fake transfer completion | Sequence validation and session binding reject unauthorized ACKs | 100% |
| Forged Resume Checkpoint | Attacker injects fraudulent resume offset to truncate data | Checkpoint cryptographic validation fails; state unchanged | 100% |
| Stale Authentication | Valid credentials presented outside temporal freshness window | Timestamp delta check fails; authentication rejected | 100% |

---

## Technical Methodology

### Optical Simulation and Atmospheric Extinction
Optical propagation through atmosphere is modeled using the Beer-Lambert law:
$$I(z) = I_0 \cdot \exp(-\gamma \cdot z)$$
where $\gamma$ denotes the extinction coefficient determined by weather condition ($\gamma_{\text{clear}} = 0.0002\text{ m}^{-1}$, $\gamma_{\text{haze}} = 0.0015\text{ m}^{-1}$, $\gamma_{\text{fog}} = 0.006\text{ m}^{-1}$, $\gamma_{\text{dense\_fog}} = 0.015\text{ m}^{-1}$). Scintillation is modeled as a log-normal irradiance distribution:
$$I = I_0 \cdot \exp\left(\sigma_R \cdot X - \frac{\sigma_R^2}{2}\right), \quad X \sim \mathcal{N}(0, 1)$$

### Coordinated Aerodynamic Banking
Aircraft heading change commands calculate physical bank angles according to turn dynamics:
$$\phi = -\arctan\left(\frac{v \cdot \dot{\psi}}{g}\right)$$
ensuring continuous, dynamically realistic flight attitude.

### Dual-Model Adaptive Kalman Tracking
The tracker runs parallel Constant Velocity (CV) and Constant Turn (CT) filters:
* **CV State**: $\mathbf{x} = [\theta_x, \theta_y, \dot{\theta}_x, \dot{\theta}_y]^T$
* **CT State**: $\mathbf{x} = [\theta_x, \theta_y, v, \psi, \omega]^T$

Innovation residuals $\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_k^-)$ and innovation covariance $\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k$ determine model likelihood:
$$\mathcal{L}_m = \frac{1}{\sqrt{(2\pi)^2 |\mathbf{S}|}} \exp\left(-\frac{1}{2} \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}\right)$$
Accumulated likelihood ratios switch the active estimator between CV (for straight cruising) and CT (during maneuvers) with hysteresis to prevent chattering.

---

## Team and Hackathon Details

* **Event**: Smart India Hackathon (SIH) 2026
* **Team Name**: Team Theseus
* **Problem Statement ID**: PS SIH26169
* **Category**: Software / Smart Automation
* **Presentation Deck**: [`docs/SIH2026-THESIUS.pdf`](docs/SIH2026-THESIUS.pdf)
* **License**: [MIT License](LICENSE)
