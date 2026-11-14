# Referencia de la API de AstroDynamics 3D

Este documento especifica las clases, funciones, tipos e interfaces públicas de **AstroDynamics 3D** en sus módulos de C++20, Python y TypeScript.

---

## 1. Biblioteca C++20 (`astrodynamics_core`)

- **Espacio de nombres:** `astrodynamics`
- **Cabecera:** [`cpp_core/include/nbody_solver.hpp`](../../cpp_core/include/nbody_solver.hpp)
- **Implementación:** [`cpp_core/src/nbody_solver.cpp`](../../cpp_core/src/nbody_solver.cpp)

### `struct Vec3`

Vector cartesiano tridimensional en doble precisión.

- **Campos:** `double x{0.0}`, `double y{0.0}`, `double z{0.0}`
- **Operadores y Métodos:**
  - `constexpr Vec3 operator+(const Vec3& o) const noexcept`
  - `constexpr Vec3 operator-(const Vec3& o) const noexcept`
  - `constexpr Vec3 operator*(double s) const noexcept`
  - `constexpr Vec3 operator/(double s) const noexcept`
  - `Vec3& operator+=(const Vec3& o) noexcept`
  - `Vec3& operator-=(const Vec3& o) noexcept`
  - `Vec3& operator*=(double s) noexcept`
  - `[[nodiscard]] constexpr double dot(const Vec3& o) const noexcept`: Producto punto $\vec{a} \cdot \vec{b}$.
  - `[[nodiscard]] constexpr Vec3 cross(const Vec3& o) const noexcept`: Producto cruz $\vec{a} \times \vec{b}$.
  - `[[nodiscard]] double norm_sq() const noexcept`: Norma euclidiana al cuadrado $\|\vec{a}\|^2$.
  - `[[nodiscard]] double norm() const noexcept`: Norma euclidiana $\|\vec{a}\|$.

### `struct Body`

Cuerpo celeste con masa y estado en el espacio de fases.

- **Campos:**
  - `std::string name{"body"}`: Identificador descriptivo.
  - `double mass{1.0}`: Masa inercial y gravitatoria ($m > 0$).
  - `Vec3 position`: Coordenadas cartesianas $\vec{r} = (x, y, z)$.
  - `Vec3 velocity`: Vector de velocidad $\vec{v} = (v_x, v_y, v_z)$.

### `enum class IntegratorType`

- `SymplecticVerlet`: Integrador Velocity Verlet simpléctico (preserva estructura hamiltoniana).
- `RungeKutta4`: Integrador Runge-Kutta clásico de 4º orden.

### `class NBodySystem`

Gestor e integrador del sistema gravitacional de $N$ cuerpos.

- **Constructor:** `explicit NBodySystem(double G = 1.0, double softening = 1e-5)`
- **Configuración del Estado:**
  - `void add_body(const Body& body)`: Agrega un cuerpo e invalida la caché de aceleraciones.
  - `void clear()`: Elimina todos los cuerpos y limpia las cachés.
  - `[[nodiscard]] std::size_t size() const noexcept`: Cantidad de cuerpos.
  - `[[nodiscard]] const std::vector<Body>& bodies() const noexcept` / `std::vector<Body>& bodies() noexcept`
  - `[[nodiscard]] double G() const noexcept` / `void set_G(double G) noexcept`
  - `[[nodiscard]] double softening() const noexcept` / `void set_softening(double eps) noexcept`
- **Observables e Invariantes Físicos:**
  - `[[nodiscard]] Vec3 center_of_mass() const noexcept`: Calcula $\vec{R}_{\text{cm}}$.
  - `[[nodiscard]] Vec3 center_of_mass_velocity() const noexcept`: Calcula $\vec{V}_{\text{cm}}$.
  - `[[nodiscard]] double total_mass() const noexcept`: Calcula $M_{\text{tot}} = \sum_{i=1}^N m_i$.
  - `[[nodiscard]] double kinetic_energy() const noexcept`: Calcula $T = \frac{1}{2} \sum_{i=1}^N m_i \|\vec{v}_i\|^2$.
  - `[[nodiscard]] double potential_energy() const noexcept`: Calcula potencial de Plummer $U$.
  - `[[nodiscard]] double total_energy() const noexcept`: Calcula $H = T + U$.
  - `[[nodiscard]] Vec3 total_angular_momentum() const noexcept`: Calcula $\vec{L} = \sum_{i=1}^N m_i (\vec{r}_i \times \vec{v}_i)$.
  - `[[nodiscard]] Vec3 total_linear_momentum() const noexcept`: Calcula $\vec{P}_{\text{tot}} = \sum_{i=1}^N m_i \vec{v}_i$.
  - `[[nodiscard]] double compute_min_distance() const noexcept`: Distancia mínima entre pares $r_{\min}$ (devuelve `1e9` si $N < 2$).
  - `[[nodiscard]] double compute_adaptive_dt(double base_dt, double eta = 0.08) const noexcept`: Paso temporal acotado al intervalo `[1e-6, base_dt]`.
  - `void reset_barycenter(bool reset_position = false) noexcept`: Anula $\vec{V}_{\text{cm}}$ y opcionalmente centra $\vec{R}_{\text{cm}}$.
- **Cálculo de Aceleraciones:**
  - `[[nodiscard]] std::vector<Vec3> compute_accelerations() const`
  - `[[nodiscard]] std::vector<Vec3> compute_accelerations(std::span<const Vec3> positions) const`
  - `void compute_accelerations_inplace(std::span<const Vec3> positions, std::span<Vec3> out_acc) const noexcept`
- **Métodos de Integración:**
  - `void step_symplectic_verlet(double dt)`: Avanza $\Delta t$ con reutilización de aceleraciones.
  - `void step_rk4(double dt)`: Avanza $\Delta t$ mediante Runge-Kutta de 4º orden.
  - `void step(double dt, IntegratorType type = IntegratorType::SymplecticVerlet)`: Despacha el paso de integración.
  - `void step_adaptive(double dt_target, IntegratorType type = IntegratorType::SymplecticVerlet, double eta = 0.08)`: Sub-pasos adaptativos hasta `MAX_SUBSTEPS = 32`.
  - `std::vector<TrajectorySnapshot> simulate(double total_time, double dt, IntegratorType type = IntegratorType::SymplecticVerlet, std::size_t snapshot_interval = 1)`: Integra la trayectoria completa devolviendo `TrajectorySnapshot` (`time`, `total_energy`, `center_of_mass`, `positions`, `velocities`).

---

## 2. Módulo Python PINN Simpléctico (`pinn_surrogate`)

Módulo: [`pinn_surrogate/model.py`](../../pinn_surrogate/model.py)

### `class HamiltonianNN(torch.nn.Module)`

Red neuronal que parametriza la función escalar de energía Hamiltoniana $H_\theta(q, p)$.

- **Parámetros del Constructor:**
  - `dim: int = 3`: Dimensión espacial por partícula.
  - `hidden_dim: int = 128`: Neuronas por capa oculta.
  - `num_layers: int = 3`: Profundidad de capas ocultas.
  - `activation: str = "tanh"`: Función de activación suave (`"tanh"` o `"silu"`).
  - `separable: bool = True`: Si es `True`, parametriza $H_\theta(q, p) = T_\theta(p) + V_\theta(q)$.
- **Métodos:**
  - `forward(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor`: Calcula la energía escalar `[..., 1]`.
  - `time_derivatives(q: torch.Tensor, p: torch.Tensor, create_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`: Obtiene $(\dot{q} = \nabla_p H_\theta, \dot{p} = -\nabla_q H_\theta)$ mediante `torch.autograd.grad`.

### `class SymplecticPINNLoss(torch.nn.Module)`

Función de pérdida simpléctica con regularización energética.

- **Parámetros del Constructor:**
  - `energy_weight: float = 0.01`: Factor $\lambda_E$.
- **Métodos:**
  - `forward(model: HamiltonianNN, q: torch.Tensor, p: torch.Tensor, q_dot_true: torch.Tensor, p_dot_true: torch.Tensor, h0_true: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]`: Evalúa la pérdida total y retorna diccionario de métricas (`loss_total`, `loss_symplectic`, `loss_q_dot`, `loss_p_dot`, `loss_energy`).

### `symplectic_euler_step(model: HamiltonianNN, q: torch.Tensor, p: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]`

Avanza $(q_n, p_n)$ un paso mediante Euler Simpléctico:

$$
p_{n+1} = p_n - \Delta t \, \nabla_q H_\theta(q_n, p_n), \qquad q_{n+1} = q_n + \Delta t \, \nabla_p H_\theta(q_n, p_{n+1})
$$

### `rollout_surrogate(model: HamiltonianNN, q0: torch.Tensor, p0: torch.Tensor, num_steps: int, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Ejecuta una trayectoria de `num_steps` pasos devolviendo `(q_trajectory, p_trajectory, h_trajectory)`.

---

## 3. Módulo Cliente Web TypeScript (`web_ui/src`)

- **Tipos:** [`web_ui/src/types.ts`](../../web_ui/src/types.ts)
- **Física:** [`web_ui/src/physics.ts`](../../web_ui/src/physics.ts)
- **Visualización:** [`web_ui/src/visuals/`](../../web_ui/src/visuals)

### Tipos Principales (`types.ts`)

- `Vector3D`: `{ x: number; y: number; z: number }`
- `CelestialBody`: `{ id: string; name: string; mass: number; radius: number; color: string; emissive?: boolean; position: Vector3D; velocity: Vector3D; trail: Vector3D[] }`
- `IntegratorType`: `'symplectic_verlet' | 'rk4' | 'pinn_surrogate'`
- `SimulationConfig`: `{ G, timeStep, softening, integrator, trailLength, paused, speedMultiplier, adaptiveTimeStep, barycenterReset, bloomEnabled, bloomIntensity }`
- `PhysicalMetrics`: `{ totalEnergy, initialEnergy, relativeEnergyError, kineticEnergy, potentialEnergy, centerOfMassSpeed, angularMomentumMagnitude, stepCount, simulatedTime, currentAdaptiveDt?, minDistance? }`

### Funciones de Física y Telemetría (`physics.ts`)

- `computeAccelerations(positions: Vector3D[], masses: number[], G: number, softening: number): Vector3D[]`
- `stepSymplecticVerlet(bodies: CelestialBody[], dt: number, G: number, softening: number): CelestialBody[]`
- `stepRK4(bodies: CelestialBody[], dt: number, G: number, softening: number): CelestialBody[]`
- `stepPINNSurrogate(bodies: CelestialBody[], dt: number, G: number, softening: number): CelestialBody[]`
- `computeMinDistance(positions: Vector3D[]): number`
- `computeAdaptiveTimeStep(bodies: CelestialBody[], G: number, baseDt: number, softening: number, eta = 0.08): number`
- `resetBarycentricDrift(bodies: CelestialBody[], resetPosition = false): CelestialBody[]`
- `computeMetrics(bodies: CelestialBody[], G: number, softening: number, initialEnergy: number, stepCount: number, simulatedTime: number, currentAdaptiveDt?: number): PhysicalMetrics`
- `PRESETS`: Sistemas orbitales predefinidos (`kepler`, `figure8`, `lagrange`).
