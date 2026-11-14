# AstroDynamics 3D System Architecture

This document specifies the software architecture and mathematical design of **AstroDynamics 3D**, a hybrid N-body gravitational simulator combining a C++20 numerical core, a symplectic Physics-Informed Neural Network (PINN) surrogate in PyTorch, and an interactive WebGL 3D client.

---

## 1. Modular Decomposition

AstroDynamics 3D is organized into three decoupled layers:

```text
+-------------------------------------------------------------------+
|                  Interactive 3D WebGL Frontend                    |
|         (React 19 + TypeScript + Three.js + Tailwind v4)          |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Symplectic PINN Surrogate (PyTorch)                 |
|     H_theta(q, p) Parametrization | Canonical Autodiff Field      |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                  C++20 High-Performance Core                      |
|      Stormer-Verlet | OpenMP | Branchless AVX2/FMA SIMD           |
+-------------------------------------------------------------------+
```

---

## 2. Layer 1: C++20 Numerical Core (`cpp_core/`)

Source files: [`cpp_core/include/nbody_solver.hpp`](../../cpp_core/include/nbody_solver.hpp), [`cpp_core/src/nbody_solver.cpp`](../../cpp_core/src/nbody_solver.cpp).

### 2.1 Gravitational Potential with Plummer Softening

Pairwise gravitational interaction is regularized using the Plummer sphere potential with softening scale $\epsilon > 0$:

$$
\Phi(r) = -\frac{G M}{\sqrt{r^2 + \epsilon^2}}
$$

The pairwise force exerted on body $i$ by body $j$ is:

$$
\vec{F}_{ij} = \frac{G m_i m_j (\vec{r}_j - \vec{r}_i)}{\left(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}
$$

### 2.2 Branchless SIMD Vectorization and SoA Layout

In standard N-body kernels, an explicit branch (`if (i == j) continue;`) skips self-interaction, disrupting SIMD vector lane execution. When $\epsilon > 0$, self-interaction yields:

$$
\vec{r}_i - \vec{r}_i = \mathbf{0} \implies \vec{a}_{ii} = \frac{G m_i \mathbf{0}}{\left(0 + \epsilon^2\right)^{3/2}} = \mathbf{0}
$$

Eliminating the conditional branch enables OpenMP SIMD reduction (`#pragma omp simd reduction(+:ax, ay, az)`) and contiguous FMA auto-vectorization over the Structure-of-Arrays (SoA) mass cache `masses_` and particle position spans. When `softening == 0.0`, the solver falls back to a guarded loop to prevent division by zero.

### 2.3 Symplectic Störmer-Verlet with Acceleration Caching

A naive Velocity Verlet implementation requires two $\mathcal{O}(N^2)$ force evaluations per time step:

1. $\vec{r}(t + \Delta t) = \vec{r}(t) + \vec{v}(t)\Delta t + \frac{1}{2}\vec{a}(t)\Delta t^2$
2. Evaluate $\vec{a}(t + \Delta t)$
3. $\vec{v}(t + \Delta t) = \vec{v}(t) + \frac{1}{2}[\vec{a}(t) + \vec{a}(t + \Delta t)]\Delta t$

`NBodySystem` maintains persistent preallocated buffers (`cur_acc_`, `scratch_pos_`, `scratch_acc_`, and RK4 stage buffers `r0_`, `v0_`, `r_scratch_`, `k1_v_`–`k4_v_`). The acceleration $\vec{a}(t+\Delta t)$ computed in step (2) is swapped into `cur_acc_` for the subsequent step, halving steady-state force evaluations ($1 \times \mathcal{O}(N^2)$ per step) and achieving zero heap allocations per step after initialization.

### 2.4 Physical Invariants and Adaptive Stepping

- **Linear Momentum & Barycentric Control:** `reset_barycenter()` enforces zero total linear momentum $\vec{P}_{\text{tot}} = \sum_{i=1}^N m_i \vec{v}_i = \mathbf{0}$ and optionally centers $\vec{R}_{\text{cm}}$.
- **Symplectic Shadow Hamiltonian:** By the Baker-Campbell-Hausdorff (BCH) expansion, Strang operator splitting preserves a modified Hamiltonian $\tilde{H} = H + \Delta t^2 H_2 + \mathcal{O}(\Delta t^4)$, bounding energy oscillations over extended integration horizons without secular drift.
- **Adaptive Sub-Stepping:** `compute_adaptive_dt()` bounds the step size relative to the shortest pairwise dynamical encounter timescale $\tau_{ij} = \sqrt{r_{ij}^3 / (G (m_i + m_j))}$.

---

## 3. Layer 2: Symplectic PINN Surrogate (`pinn_surrogate/`)

Source file: [`pinn_surrogate/model.py`](../../pinn_surrogate/model.py).

### 3.1 Hamiltonian Neural Network Parametrization

Rather than predicting Cartesian accelerations directly (which permits non-conservative drift), `HamiltonianNN` parametrizes the scalar Hamiltonian energy surface $H_\theta(q, p): \mathbb{R}^{2d} \to \mathbb{R}$. When `separable=True`, it decomposes energy into kinetic and potential branches:

$$
H_\theta(q, p) = T_\theta(p) + V_\theta(q)
$$

Canonical phase space vector fields are obtained via PyTorch autograd:

$$
\hat{\dot{q}} = \frac{\partial H_\theta}{\partial p}, \qquad \hat{\dot{p}} = -\frac{\partial H_\theta}{\partial q}
$$

### 3.2 Symplectic Loss Formulation

`SymplecticPINNLoss` enforces Hamilton's canonical equations while optionally anchoring the Hamiltonian energy gauge against $H_0$:

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{k=1}^B \left( \left\|\frac{\partial H_\theta}{\partial p} - \dot{q}_k\right\|_2^2 + \left\|\frac{\partial H_\theta}{\partial q} + \dot{p}_k\right\|_2^2 \right) + \lambda_E \frac{1}{B}\sum_{k=1}^B \|H_\theta(q_k, p_k) - H_0\|^2
$$

Smooth $C^\infty$ activation functions (`Tanh` or `SiLU`) ensure well-defined second-order mixed derivatives during training.

### 3.3 Symplectic Euler Rollout

`symplectic_euler_step()` and `rollout_surrogate()` propagate trajectories with a semi-implicit symplectic Euler step:

$$
\begin{aligned}
p_{n+1} &= p_n - \Delta t \left.\frac{\partial H_\theta}{\partial q}\right|_{(q_n, p_n)} \\
q_{n+1} &= q_n + \Delta t \left.\frac{\partial H_\theta}{\partial p}\right|_{(q_n, p_{n+1})}
\end{aligned}
$$

For separable Hamiltonians, the Jacobian determinant satisfies $\det J = 1$, preserving the canonical symplectic 2-form $\omega = dq \wedge dp$ and Liouville phase space volume.

---

## 4. Layer 3: Interactive 3D WebGL Frontend (`web_ui/`)

Source directory: [`web_ui/src/`](../../web_ui/src).

### 4.1 Rendering Pipeline and Scene Graph

- **Stack:** React 19, TypeScript, Vite 6, Tailwind CSS v4, and Three.js.
- **Post-Processing:** `EffectComposer` chaining `RenderPass` and `UnrealBloomPass` with `ACESFilmicToneMapping` (exposure `1.1`, threshold `0.18`).
- **Scene Components:**
  - **Cosmic Starfield** ([`CosmicStarfield.ts`](../../web_ui/src/visuals/CosmicStarfield.ts)): 5,500 procedural star particles sampled across Harvard spectral classes (O through M) with Gaussian point shaders, accompanied by 6 volumetric nebula dust planes.
  - **Solar & Atmospheric Shaders** ([`SunShader.ts`](../../web_ui/src/visuals/SunShader.ts), [`AtmosphereShader.ts`](../../web_ui/src/visuals/AtmosphereShader.ts)): 3D Simplex noise photospheric granulation and corona pulsation for emissive stars; inverted-Fresnel Rayleigh scattering shells for planetary bodies.
  - **Orbital Trails** ([`GlowingTrail.ts`](../../web_ui/src/visuals/GlowingTrail.ts)): Dynamic `THREE.Line` geometry with vertex-alpha tail attenuation.
  - **Reference Grid:** 50x50 subdivision `GridHelper` aligned on the ecliptic plane.
- **Resource Lifecycle:** `disposeHierarchy()` explicitly disposes geometries, materials, shader passes, and WebGL context on unmount, verified leak-free via [`verify_webgl_disposal.cjs`](../../web_ui/scripts/verify_webgl_disposal.cjs).

### 4.2 Simulation Controls and Live Telemetry

- **Client Physics Engine** ([`physics.ts`](../../web_ui/src/physics.ts)): Implements `stepSymplecticVerlet`, `stepRK4`, `stepPINNSurrogate`, adaptive sub-stepping (`computeAdaptiveTimeStep`), barycentric drift correction (`resetBarycentricDrift`), and astrophysical presets (`kepler`, `figure8`, `lagrange`).
- **Mission Deck** ([`ControlsPanel.tsx`](../../web_ui/src/components/ControlsPanel.tsx)): Configures $G$, $\Delta t$, integrator scheme, trail length, HDR bloom, per-body mass/velocity state, and displays throttled (~15 Hz) invariant telemetry ($H$, $\Delta H / |H_0|$, $\|\vec{V}_{\text{cm}}\|$, $\|\vec{L}\|$, $r_{\min}$) with a logarithmic error sparkline.
