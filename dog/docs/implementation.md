# Quadruped Simulation & RL Implementation Plan

This document outlines the step-by-step implementation plan for simulating the quadruped robot (`dog`), training a reinforcement learning (RL) locomotion policy, and running real-time co-simulation with joint-level physics and energy efficiency telemetry.

---

## Architecture Overview

The system operates across two synchronised loops:
1. **High-Level Control / RL Policy Loop (50 Hz / 20 ms)**: Observes 34 robot state variables and emits 12 target joint angle offsets.
2. **Low-Level Simulation & PD Control Loop (1000 Hz / 1 ms)**: MuJoCo physics engine executes 20 substeps per control action, running high-frequency PD torque controllers on all 12 joints:
   $$\tau_i = \text{clip}(K_p (q_i^* - q_i) - K_d \dot{q}_i, \tau_{\min}, \tau_{\max})$$

```
                   ┌──────────────────────────────────────┐
                   │    RL Policy Network (Actor-Critic)  │
                   └───────────┬──────────────────▲───────┘
          12 Desired Angles    │                  │  34 Robot State Observations
          (Action offsets)     │                  │  (Positions, Velocities, Quat)
                   ┌───────────▼──────────────────┴───────┐
                   │   QuadrupedEnv (Gymnasium Env)       │
                   └───────────┬──────────────────▲───────┘
          PD Torques           │                  │  Contacts & Kinematic States
          (20 Substeps @ 1kHz) │                  │  (Foot forces, chassis pose)
                   ┌───────────▼──────────────────┴───────┐
                   │    MuJoCo Physics Engine (dog.xml)   │
                   └──────────────────┬───────────────────┘
                                      │ Synchronous Frame Sync (50 FPS)
                   ┌──────────────────▼───────────────────┐
                   │       mujoco.viewer (Passive UI)     │
                   │    Live Performance & CoT Telemetry   │
                   └──────────────────────────────────────┘
```

---

## Phase 1: Environment & Dependency Setup

### 1.1 Python Environment
Establish an isolated virtual environment with the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mujoco gymnasium numpy scipy matplotlib stable-baselines3 torch
```

### 1.2 Model Verification
Ensure [dog.xml](file:///Users/alex/Downloads/robotics/dog/dog.xml) loads cleanly in MuJoCo and keyframes initialize without geometry penetrations:
- Verify ground clearance: standing foot z-level should rest tangentially on `floor` at $z = -0.18\,\text{m}$.
- Confirm actuator limits (`ctrlrange="-1 1"`) and joint ranges.
- Validate robot reference dimensions: links $L_1 = 0.1\,\text{m}, L_2 = 0.1\,\text{m}$ defined in [robot_structure.py](file:///Users/alex/Downloads/robotics/dog/robot_structure.py).

---

## Phase 2: Complete the Gymnasium Environment ([QuadrupedEnv.py](file:///Users/alex/Downloads/robotics/dog/QuadrupedEnv.py))

Update [QuadrupedEnv.py](file:///Users/alex/Downloads/robotics/dog/QuadrupedEnv.py) from a passive stub into a standard, vectorized-ready `gym.Env`.

### 2.1 Complete the Physics Step Loop
In `step(action)`:
- Compute target joint angles: $q^* = q_{\text{current}} + \alpha \cdot a$, where $\alpha = 0.25\,\text{rad}$.
- Run 20 substeps ($20 \times 0.001\,\text{s} = 0.02\,\text{s}$):
  - Compute PD torques for all 12 motors with gains $K_p = 5.0, K_d = 0.15$.
  - Clamp torque output to actuator limits $[-1.0, 1.0]\,\text{Nm}$.
  - Accumulate electrical energy expenditure: $E_{\text{elec}} = \sum \tau_i^2 \Delta t$.
  - Accumulate mechanical work: $W_{\text{mech}} = \sum |\tau_i \dot{q}_i| \Delta t$.
  - Advance physics state using `mujoco.mj_step(self.model, self.data)`.

### 2.2 Multi-Objective Reward Function
Balance locomotion performance against energy consumption:
$$R = R_{\text{velocity}} + R_{\text{survival}} + R_{\text{orientation}} + R_{\text{energy}} + R_{\text{action\_smoothness}}$$

| Component | Mathematical Term | Purpose |
| :--- | :--- | :--- |
| **Velocity Tracking** | $-c_v |v_x - v_x^*|$ or $\exp(-\frac{(v_x - v_x^*)^2}{\sigma^2})$ | Encourages moving forward at target speed ($v_x^* = 0.5\,\text{m/s}$) |
| **Survival Bonus** | $+1.0$ per alive step | Prevents premature collapse |
| **Chassis Orientation** | $-c_{\text{ori}} (1.0 - \mathbf{z}_{\text{chassis}} \cdot \mathbf{e}_z)$ | Keeps torso level (penalizes pitch and roll) |
| **Lateral Drift** | $-c_y |v_y| - c_\omega |\omega_z|$ | Penalizes sideways sliding and turning |
| **Energy Penalty** | $-c_{\tau} \sum \tau_i^2 - c_w \sum |\tau_i \dot{q}_i|$ | Minimizes motor heating and mechanical work (improves efficiency) |
| **Action Smoothness** | $-c_{\Delta} \|a_t - a_{t-1}\|^2$ | Suppresses high-frequency joint jitter |

### 2.3 Termination & Truncation Rules
- **Termination (`terminated = True`)**:
  - Fall detection: Chassis center-of-mass height $z_{\text{chassis}} < 0.08\,\text{m}$ (nominal height $\approx 0.16\,\text{m}$).
  - Inversion/Tip-over: Vertical component of chassis rotation matrix $R_{zz} < 0.5$ (tilt $> 60^\circ$).
  - Body contact: Floor collision detected on `chassis_geom` or hip spheres.
- **Truncation (`truncated = True`)**:
  - Episode time limit reached (e.g. $T_{\max} = 1000$ steps $= 20\,\text{seconds}$).

### 2.4 Built-in Viewer Support
Support `render_mode="human"`:
- Maintain a passive viewer handle (`self.viewer`) via `mujoco.viewer.launch_passive`.
- Synchronize physics buffer with render buffer on `render()` calls.

---

## Phase 3: Energy Efficiency & Performance Telemetry

Implement real-time diagnostics to evaluate how effectively and efficiently the robot runs.

### 3.1 Cost of Transport (CoT)
Dimensionless metric measuring energy consumption per unit weight and distance:
$$\text{CoT} = \frac{E_{\text{total}}}{m \cdot g \cdot d}$$
- $E_{\text{total}}$: Total mechanical work $\int \sum |\tau_i \dot{q}_i| dt$ or electrical power proxy $\int \sum \tau_i^2 dt$.
- $m$: Total robot mass ($\approx 0.72\,\text{kg}$ as modeled in [dog.xml](file:///Users/alex/Downloads/robotics/dog/dog.xml)).
- $g$: Gravity constant ($9.81\,\text{m/s}^2$).
- $d$: Distance traveled along the x-axis ($x(t) - x(0)$).
- **Target**: Natural quadruped biological gaits achieve $\text{CoT} \approx 0.4 - 1.5$; unoptimized robotic policies often start at $> 5.0$.

### 3.2 Performance Metrics
- **Mean Velocity Error**: $\text{RMSE}_v = \sqrt{\frac{1}{N} \sum (v_x - v_x^*)^2}$.
- **Posture Stability**: Variance in torso pitch, roll, and height.
- **Real-Time Factor (RTF)**: $\text{RTF} = \frac{t_{\text{simulated}}}{t_{\text{wall-clock}}}$.
- **Joint Jitter**: Mean absolute joint jerk $\frac{d^2 \dot{q}}{dt^2}$.

---

## Phase 4: Training Pipeline (Headless Execution)

Create `train_dog_rl.py` for headless, parallel training.

### 4.1 Algorithm Selection
- **Primary Algorithm**: **PPO (Proximal Policy Optimization)**:
  - Best stability for continuous quadruped action spaces with PD target offsets.
  - Policy network: 2 hidden layers $[128, 128]$ with Tanh/ReLU activations.
  - Action output: Mean $\mu \in \mathbb{R}^{12}$, log std $\sigma \in \mathbb{R}^{12}$.
- **Alternative / First-Principles**: Integrate custom multi-layer perceptron from [basic_neural_network.py](file:///Users/alex/Downloads/robotics/models_from_scratch/basic_neural_network.py) for forward policy inference and value estimation.

### 4.2 Vectorized Environments & Hardware Acceleration
- Use `gymnasium.vector.AsyncVectorEnv` or Stable-Baselines3 `SubprocVecEnv` with 8–16 parallel environments.
- Target throughput: $> 5,000$ environment steps per second headless on Apple Silicon.
- Training duration: $\sim 2 - 5\text{M}$ steps to achieve steady trotting/walking.

### 4.3 Curriculum Strategy
1. **Stage 1 (Stance & Height)**: Target velocity $v_x^* = 0\,\text{m/s}$ (learn standing balance and reject disturbances).
2. **Stage 2 (Slow Walk)**: Target velocity $v_x^* = 0.2\,\text{m/s}$ with higher alive bonus.
3. **Stage 3 (Trotting & Efficiency)**: Target velocity $v_x^* = 0.5\,\text{m/s}$ with full energy penalty active to force natural gait coordination.

---

## Phase 5: Live Co-Simulation & Interactive Evaluation

Create `simulate_dog_rl.py` to run the trained model inside the interactive visualizer.

### 5.1 Real-Time Synchronization
- Launch `mujoco.viewer.launch_passive(model, data)`.
- Enforce wall-clock synchronization ($50\,\text{FPS}$ / $20\,\text{ms}$ per control frame) using high-resolution timers (`time.perf_counter()`).
- Support pausing, resetting, and applying interactive perturbation forces via viewer mouse drags.

### 5.2 Live Telemetry HUD / Console Overlay
Print/display live metrics at 2 Hz:
```text
======================= LIVE TELEMETRY =======================
Sim Time: 12.4s | FPS: 49.8 | RTF: 1.00x
Forward Vel: 0.48 m/s (Target: 0.50 m/s) | Error: 0.02 m/s
Chassis Height: 0.158 m | Pitch: 1.2° | Roll: -0.4°
Instantaneous Power: 2.14 W | Accumulated Energy: 24.8 J
Current CoT: 0.82 (High Efficiency)
Foot Contacts: [FL: Yes | FR: No | RL: No | RR: Yes]
==============================================================
```

---

## Execution Checklist

- [x] **Step 1**: Install environment dependencies (`mujoco`, `gymnasium`, `stable-baselines3`, `torch`).
- [x] **Step 2**: Implement full `step()`, `reward()`, `_is_terminated()`, and `render()` in [QuadrupedEnv.py](file:///Users/alex/Downloads/robotics/dog/QuadrupedEnv.py).
- [x] **Step 3**: Validate [QuadrupedEnv.py](file:///Users/alex/Downloads/robotics/dog/QuadrupedEnv.py) with random actions and PD controller sanity check.
- [ ] **Step 4**: Implement `train_dog_rl.py` (PPO training script with checkpoint saving and TensorBoard logging).
- [ ] **Step 5**: Implement `simulate_dog_rl.py` (loads trained weights, runs passive MuJoCo viewer with live CoT and performance telemetry).
- [ ] **Step 6**: Benchmark trained policy vs. kinematic trajectories from [dog_one_leg_up.py](file:///Users/alex/Downloads/robotics/dog/dog_one_leg_up.py) and [leg_fixed_4.py](file:///Users/alex/Downloads/robotics/dog/leg_fixed_4.py).
