# AstroDynamics 3D API Reference

This document specifies the public classes, functions, types, and interfaces of **AstroDynamics 3D** across its C++20, Python, and TypeScript modules.

---

## 1. C++20 Core Library (`astrodynamics_core`)

- **Namespace:** `astrodynamics`
- **Header:** [`cpp_core/include/nbody_solver.hpp`](../../cpp_core/include/nbody_solver.hpp)
- **Implementation:** [`cpp_core/src/nbody_solver.cpp`](../../cpp_core/src/nbody_solver.cpp)

### `struct Vec3`

Double-precision 3D Cartesian vector.

- **Fields:** `double x{0.0}`, `double y{0.0}`, `double z{0.0}`
- **Operators and Methods:**
  - `constexpr Vec3 operator+(const Vec3& o) const noexcept`
  - `constexpr Vec3 operator-(const Vec3& o) const noexcept`
  - `constexpr Vec3 operator*(double s) const noexcept`
  - `constexpr Vec3 operator/(double s) const noexcept`
  - `Vec3& operator+=(const Vec3& o) noexcept`
  - `Vec3& operator-=(const Vec3& o) noexcept`
  - `Vec3& operator*=(double s) noexcept`
  - `[[nodiscard]] constexpr double dot(const Vec3& o) const noexcept`: Dot product $\vec{a} \cdot \vec{b}$.
  - `[[nodiscard]] constexpr Vec3 cross(const Vec3& o) const noexcept`: Cross product $\vec{a} \times \vec{b}$.
  - `[[nodiscard]] double norm_sq() const noexcept`: Squared Euclidean norm $\|\vec{a}\|^2$.
  - `[[nodiscard]] double norm() const noexcept`: Euclidean norm $\|\vec{a}\|$.

### `struct Body`

Astronomical body with mass and phase space coordinates.

- **Fields:**
  - `std::string name{"body"}`: Descriptive identifier.
  - `double mass{1.0}`: Inertial and gravitational mass ($m > 0$).
  - `Vec3 position`: Cartesian coordinates $\vec{r} = (x, y, z)$.
  - `Vec3 velocity`: Velocity vector $\vec{v} = (v_x, v_y, v_z)$.

### `enum class IntegratorType`

- `SymplecticVerlet`: Symplectic Störmer-Verlet integrator (Hamiltonian-preserving).
- `RungeKutta4`: Classical 4th-order Runge-Kutta integrator.

### `class NBodySystem`

N-body gravitational system manager and integrator.

- **Constructor:** `explicit NBodySystem(double G = 1.0, double softening = 1e-5)`
- **State Configuration:**
  - `void add_body(const Body& body)`: Appends a body and invalidates cached accelerations.
  - `void clear()`: Clears all bodies and mass/acceleration caches.
  - `[[nodiscard]] std::size_t size() const noexcept`: Number of bodies.
  - `[[nodiscard]] const std::vector<Body>& bodies() const noexcept` / `std::vector<Body>& bodies() noexcept`
  - `[[nodiscard]] double G() const noexcept` / `void set_G(double G) noexcept`
  - `[[nodiscard]] double softening() const noexcept` / `void set_softening(double eps) noexcept`
- **Observables and Physical Invariants:**
  - `[[nodiscard]] Vec3 center_of_mass() const noexcept`: Computes $\vec{R}_{\text{cm}}$.
  - `[[nodiscard]] Vec3 center_of_mass_velocity() const noexcept`: Computes $\vec{V}_{\text{cm}}$.
  - `[[nodiscard]] double total_mass() const noexcept`: Computes $M_{\text{tot}} = \sum_{i=1}^N m_i$.
  - `[[nodiscard]] double kinetic_energy() const noexcept`: Computes $T = \frac{1}{2} \sum_{i=1}^N m_i \|\vec{v}_i\|^2$.
  - `[[nodiscard]] double potential_energy() const noexcept`: Computes Plummer potential $U$.
  - `[[nodiscard]] double total_energy() const noexcept`: Computes $H = T + U$.
  - `[[nodiscard]] Vec3 total_angular_momentum() const noexcept`: Computes $\vec{L} = \sum_{i=1}^N m_i (\vec{r}_i \times \vec{v}_i)$.
  - `[[nodiscard]] Vec3 total_linear_momentum() const noexcept`: Computes $\vec{P}_{\text{tot}} = \sum_{i=1}^N m_i \vec{v}_i$.
  - `[[nodiscard]] double compute_min_distance() const noexcept`: Minimum pairwise distance $r_{\min}$ (returns `1e9` if $N < 2$).
  - `[[nodiscard]] double compute_adaptive_dt(double base_dt, double eta = 0.08) const noexcept`: Encounter timescale step bound clamped to `[1e-6, base_dt]`.
  - `void reset_barycenter(bool reset_position = false) noexcept`: Zeroes $\vec{V}_{\text{cm}}$ and optionally $\vec{R}_{\text{cm}}$.
- **Acceleration Kernels:**
  - `[[nodiscard]] std::vector<Vec3> compute_accelerations() const`
  - `[[nodiscard]] std::vector<Vec3> compute_accelerations(std::span<const Vec3> positions) const`
  - `void compute_accelerations_inplace(std::span<const Vec3> positions, std::span<Vec3> out_acc) const noexcept`
- **Integration Methods:**
  - `void step_symplectic_verlet(double dt)`: Advances state by $\Delta t$ with cached accelerations.
  - `void step_rk4(double dt)`: Advances state by $\Delta t$ via 4th-order Runge-Kutta.
  - `void step(double dt, IntegratorType type = IntegratorType::SymplecticVerlet)`: Dispatches integration step.
  - `void step_adaptive(double dt_target, IntegratorType type = IntegratorType::SymplecticVerlet, double eta = 0.08)`: Adaptive sub-stepping up to `MAX_SUBSTEPS = 32`.
  - `std::vector<TrajectorySnapshot> simulate(double total_time, double dt, IntegratorType type = IntegratorType::SymplecticVerlet, std::size_t snapshot_interval = 1)`: Multi-step rollout returning `TrajectorySnapshot` (`time`, `total_energy`, `center_of_mass`, `positions`, `velocities`).

---

## 2. Python Symplectic PINN Module (`pinn_surrogate`)

Module: [`pinn_surrogate/model.py`](../../pinn_surrogate/model.py)

### `class HamiltonianNN(torch.nn.Module)`

Neural network parametrizing scalar Hamiltonian energy $H_\theta(q, p)$.

- **Constructor Parameters:**
  - `dim: int = 3`: Coordinate dimensionality per particle.
  - `hidden_dim: int = 128`: Hidden layer width.
  - `num_layers: int = 3`: Hidden layer depth.
  - `activation: str = "tanh"`: Activation function (`"tanh"` or `"silu"`).
  - `separable: bool = True`: If `True`, parametrizes $H_\theta(q, p) = T_\theta(p) + V_\theta(q)$.
- **Methods:**
  - `forward(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor`: Evaluates scalar Hamiltonian energy `[..., 1]`.
  - `time_derivatives(q: torch.Tensor, p: torch.Tensor, create_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`: Computes canonical derivatives $(\dot{q} = \nabla_p H_\theta, \dot{p} = -\nabla_q H_\theta)$ via `torch.autograd.grad`.

### `class SymplecticPINNLoss(torch.nn.Module)`

Symplectic canonical field residual and energy conservation loss.

- **Constructor Parameters:**
  - `energy_weight: float = 0.01`: Regularization weight $\lambda_E$.
- **Methods:**
  - `forward(model: HamiltonianNN, q: torch.Tensor, p: torch.Tensor, q_dot_true: torch.Tensor, p_dot_true: torch.Tensor, h0_true: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]`: Computes total loss and returns component dictionary (`loss_total`, `loss_symplectic`, `loss_q_dot`, `loss_p_dot`, `loss_energy`).

### `symplectic_euler_step(model: HamiltonianNN, q: torch.Tensor, p: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]`

Advances $(q_n, p_n)$ by one discrete Symplectic Euler step:

$$
p_{n+1} = p_n - \Delta t \, \nabla_q H_\theta(q_n, p_n), \qquad q_{n+1} = q_n + \Delta t \, \nabla_p H_\theta(q_n, p_{n+1})
$$

### `rollout_surrogate(model: HamiltonianNN, q0: torch.Tensor, p0: torch.Tensor, num_steps: int, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Rolls out `num_steps` integration steps and returns `(q_trajectory, p_trajectory, h_trajectory)`.

---

## 3. TypeScript Web Client Module (`web_ui/src`)

- **Types:** [`web_ui/src/types.ts`](../../web_ui/src/types.ts)
- **Physics:** [`web_ui/src/physics.ts`](../../web_ui/src/physics.ts)
- **Visuals:** [`web_ui/src/visuals/`](../../web_ui/src/visuals)

### Core Types (`types.ts`)

- `Vector3D`: `{ x: number; y: number; z: number }`
- `CelestialBody`: `{ id: string; name: string; mass: number; radius: number; color: string; emissive?: boolean; position: Vector3D; velocity: Vector3D; trail: Vector3D[] }`
- `IntegratorType`: `'symplectic_verlet' | 'rk4' | 'pinn_surrogate'`
- `SimulationConfig`: `{ G, timeStep, softening, integrator, trailLength, paused, speedMultiplier, adaptiveTimeStep, barycenterReset, bloomEnabled, bloomIntensity }`
- `PhysicalMetrics`: `{ totalEnergy, initialEnergy, relativeEnergyError, kineticEnergy, potentialEnergy, centerOfMassSpeed, angularMomentumMagnitude, stepCount, simulatedTime, currentAdaptiveDt?, minDistance? }`

### Physics & Telemetry Functions (`physics.ts`)

- `computeAccelerations(positions: Vector3D[], masses: number[], G: number, softening: number): Vector3D[]`
- `stepSymplecticVerlet(bodies: CelestialBody[], dt: number, G: number, softening: number): CelestialBody[]`
- `stepRK4(bodies: CelestialBody[], dt: number, G: number, softening: number): CelestialBody[]`
- `stepPINNSurrogate(bodies: CelestialBody[], dt: number, G: number, softening: number): CelestialBody[]`
- `computeMinDistance(positions: Vector3D[]): number`
- `computeAdaptiveTimeStep(bodies: CelestialBody[], G: number, baseDt: number, softening: number, eta = 0.08): number`
- `resetBarycentricDrift(bodies: CelestialBody[], resetPosition = false): CelestialBody[]`
- `computeMetrics(bodies: CelestialBody[], G: number, softening: number, initialEnergy: number, stepCount: number, simulatedTime: number, currentAdaptiveDt?: number): PhysicalMetrics`
- `PRESETS`: Pre-configured orbital systems (`kepler`, `figure8`, `lagrange`).
