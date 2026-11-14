# AstroDynamics 3D: N-Body Gravitational Engine & Symplectic PINN Surrogate

[![Status: v1.0.0 - Archived](https://img.shields.io/badge/Status-v1.0.0_--_Archived-lightgrey.svg)](#)
[![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](#)
[![Version: Spanish](https://img.shields.io/badge/Version-Espa%C3%B1ol-green.svg)](README.es.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19.0+-61dafb.svg)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-r173+-black.svg)](https://threejs.org/)

> **High-performance celestial mechanics integration engine with symplectic Physics-Informed Neural Networks (PINNs) and interactive 3D WebGL visualization.**
>
> Author: **Moisés Amundarain** | Read in Spanish: [README.es.md](README.es.md)

> **Archived Project (`v1.0.0 - Archived`):** This repository is feature-complete and archived for reference.

---

## Mathematical and Physical Foundations

### 1. N-Body Celestial Mechanics with Plummer Softening

Pairwise gravitational interactions are regularized using the Plummer sphere potential with softening scale $\epsilon > 0$ to avoid close-encounter singularities:

$$
U(r_1, \dots, r_N) = -\sum_{1 \le i < j \le N} \frac{G m_i m_j}{\sqrt{\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2}}
$$

$$
\vec{a}_i = -\frac{1}{m_i} \nabla_{\vec{r}_i} U = \sum_{j \neq i} \frac{G m_j (\vec{r}_j - \vec{r}_i)}{\left(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}
$$

**Branchless SIMD Evaluation:** For self-interaction ($j = i$), $(\vec{r}_i - \vec{r}_i) = \mathbf{0}$, so the force contribution evaluates identically to $\mathbf{0}$ when $\epsilon > 0$. This removes conditional branching inside the inner loop, allowing contiguous AVX2/FMA auto-vectorization and OpenMP SIMD reductions.

### 2. Symplectic Geometry and Invariant Conservation

The phase space $\mathcal{M} = \mathbb{R}^{2d}$ carries the canonical symplectic 2-form:

$$
\omega = \sum_{k=1}^d dq_k \wedge dp_k
$$

- **Exact Symplecticity:** The discrete flow $\Phi_{\Delta t}$ preserves the differential 2-form: $\Phi_{\Delta t}^* \omega = \omega$.
- **Liouville Phase Space Volume:** Preservation of $\Omega = \bigwedge^d \omega$ implies $\det\left(\frac{\partial(q_{n+1}, p_{n+1})}{\partial(q_n, p_n)}\right) = 1$.
- **Shadow Hamiltonian (BCH Expansion):** Symplectic Velocity Verlet preserves a modified shadow Hamiltonian $\tilde{H}(q, p) = H(q, p) + \Delta t^2 H_2(q, p) + \mathcal{O}(\Delta t^4)$, bounding energy oscillations without secular drift over extended integrations.

### 3. Hamiltonian Neural Networks (HNN) and Symplectic PINNs

Instead of predicting Cartesian accelerations directly, the PyTorch surrogate parametrizes the scalar Hamiltonian $H_\theta(q, p)$:

$$
\dot{q} = \frac{\partial H_\theta}{\partial p}, \qquad \dot{p} = -\frac{\partial H_\theta}{\partial q}
$$

The symplectic training objective enforces Hamilton's canonical equations via autograd alongside energy anchoring:

$$
\mathcal{L}_{\text{total}}(\theta) = \left\| \frac{\partial H_\theta}{\partial p} - \dot{q}_{\text{true}} \right\|^2 + \left\| \frac{\partial H_\theta}{\partial q} + \dot{p}_{\text{true}} \right\|^2 + \lambda_E \|H_\theta(q_t, p_t) - H_0\|^2
$$

---

## Documentation

Full technical specifications and theoretical guides are organized under [`docs/`](docs/README.md):

| Category | Document | Description |
| :--- | :--- | :--- |
| **Index** | [docs/README.md](docs/README.md) | Master documentation index |
| **Wiki (Reference)** | [docs/wiki/ARCHITECTURE.md](docs/wiki/ARCHITECTURE.md) ([ES](docs/wiki/ARCHITECTURE.es.md)) | Three-layer system design, solvers, PINN architecture, and WebGL pipeline |
| **Wiki (Reference)** | [docs/wiki/API_REFERENCE.md](docs/wiki/API_REFERENCE.md) ([ES](docs/wiki/API_REFERENCE.es.md)) | Public APIs for `cpp_core`, `pinn_surrogate`, and `web_ui` |
| **Notes (Theory)** | [docs/notes/01_orbital_mechanics_guide.md](docs/notes/01_orbital_mechanics_guide.md) | Newtonian N-body dynamics, Plummer softening, RK4 vs. Störmer-Verlet (ES) |
| **Notes (Theory)** | [docs/notes/02_pinns_and_hnn_guide.md](docs/notes/02_pinns_and_hnn_guide.md) | Hamiltonian formulations, symplectic loss, and Liouville proof (ES) |
| **Notes (Theory)** | [docs/notes/03_hpc_architecture_guide.md](docs/notes/03_hpc_architecture_guide.md) | SoA memory layout, zero-allocation buffers, and OpenMP scaling (ES) |
| **Notes (Theory)** | [docs/notes/04_webgl_shaders_guide.md](docs/notes/04_webgl_shaders_guide.md) | Rayleigh glow, Simplex solar granulation, starfield sampling, and ACES HDR (ES) |

---

## Verification and Test Suites

| Component | Suite / Script | Scope Verified |
| :--- | :--- | :--- |
| **C++20 Core** | GoogleTest / CTest (`test_nbody`) | 10 tests: center-of-mass conservation, Verlet and RK4 energy bounds, angular momentum, adaptive time-stepping, barycentric reset, zero-softening safety |
| **C++20 Memory & Scaling** | Benchmark (`benchmark_hpc`) | Steady-state `0` heap allocations per step across $N \in \{2, 3, 128, 512, 1024\}$ for Verlet and RK4; OpenMP multi-thread throughput |
| **Symplectic PINN** | Pytest (`test_pinn_conservation.py`) | 4 tests: autograd finite-difference gradient check, symplectic loss backpropagation, 1,000-step bounded energy conservation, separable HNN stability |
| **Web UI Build & Disposal** | TypeScript + Vite + Node (`verify_webgl_disposal.cjs`) | Clean `tsc -b && vite build`; 500-cycle mount/unmount GPU geometry/material disposal audit |
| **Browser E2E (Optional)** | Playwright (`e2e_browser_test_astrodynamics.cjs`) | Interactive preset switching, Invariants/Bodies tabs, and zero uncaught page errors against `http://localhost:5180` |

---

## Quick Start

### Prerequisites

- **C++ Compiler:** GCC 13+ or Clang 17+ with C++20 support and OpenMP (`libomp-dev` / `libgomp`).
- **Build System:** CMake 3.20+.
- **Python:** Python 3.10+ (`3.11` or `3.12` recommended) with `venv` / `pip`.
- **Node.js:** Node.js 18+ or 20+ with `npm`.

### 1. Build and Test the C++20 Numerical Core

```bash
cmake -B cpp_core/build -S cpp_core -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_core/build -j$(nproc)
ctest --test-dir cpp_core/build --output-on-failure

# Optional HPC throughput and zero-allocation audit
./cpp_core/build/benchmark_hpc
```

### 2. Run the Symplectic PINN Surrogate

```bash
cd pinn_surrogate
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU), NumPy, and pytest via pyproject.toml
pip install -e ".[test]"

# Run unit and conservation tests
pytest tests/ -v

# Run Hamiltonian rollout inference demo and benchmark
python3 demo_inference.py
python3 benchmark_pinn.py
```

### 3. Launch the 3D WebGL Frontend

```bash
cd web_ui
npm install

# Start Vite development server on port 5180
npm run dev

# Or build and preview production bundle
npm run build
npm run preview
```

Open `http://localhost:5180` in a WebGL-capable browser.

**Interactive Controls:**
- **Left-Click + Drag:** Orbit 3D camera.
- **Right-Click + Drag:** Pan camera.
- **Mouse Wheel:** Zoom.
- **Mission Deck:**
  - **Physics Tab:** Switch presets (`kepler`, `figure8`, `lagrange`), toggle adaptive time-stepping, barycentric lock, and HDR bloom, adjust $G$ and $\Delta t$, and select between `Symplectic Verlet`, `Runge-Kutta 4th Order`, and `Symplectic PINN Surrogate`.
  - **Bodies Tab:** Modify individual celestial body mass $m$ and velocity components $(v_x, v_y, v_z)$.
  - **Invariants Tab:** Monitor $H$, $H_0$, $\Delta H / |H_0|$ with live logarithmic sparkline, $\|\vec{V}_{\text{cm}}\|$, $\|\vec{L}\|$, $r_{\min}$, and effective $\Delta t$.

**Optional WebGL Audits:**

```bash
# GPU resource disposal stress audit (500 cycles)
node --expose-gc web_ui/scripts/verify_webgl_disposal.cjs

# Playwright E2E test (requires `npm run dev` running on port 5180)
node web_ui/scripts/e2e_browser_test_astrodynamics.cjs
```

---

## Repository Structure

```text
astrodynamics-3d/
├── cpp_core/
│   ├── include/
│   │   └── nbody_solver.hpp                  # NBodySystem, Vec3, Body, IntegratorType
│   ├── src/
│   │   └── nbody_solver.cpp                  # SoA cache, OpenMP SIMD, Verlet, RK4, adaptive dt
│   ├── tests/
│   │   ├── test_nbody.cpp                    # GoogleTest physical invariant suite (10 tests)
│   │   └── benchmark_hpc.cpp                 # Scaling and zero-allocation heap benchmark
│   └── CMakeLists.txt                        # C++20, -O3, -march=native, OpenMP, GoogleTest
├── pinn_surrogate/
│   ├── tests/
│   │   └── test_pinn_conservation.py         # Pytest autograd & 1,000-step conservation tests
│   ├── benchmark_pinn.py                     # Latency and memory benchmark for HNN rollout
│   ├── demo_inference.py                     # CLI Hamiltonian rollout conservation demo
│   ├── model.py                              # HamiltonianNN, SymplecticPINNLoss, Euler rollout
│   └── pyproject.toml                        # Python package metadata and dependencies
├── web_ui/
│   ├── scripts/
│   │   ├── e2e_browser_test_astrodynamics.cjs # Playwright E2E browser verification
│   │   └── verify_webgl_disposal.cjs         # Three.js disposal and heap leak stress test
│   ├── src/
│   │   ├── components/
│   │   │   ├── ControlsPanel.tsx             # Interactive controls and Invariants HUD
│   │   │   └── OrbitCanvas3D.tsx             # Three.js scene, EffectComposer, bloom, loop
│   │   ├── visuals/
│   │   │   ├── AtmosphereShader.ts           # Inverted-Fresnel Rayleigh atmospheric halo
│   │   │   ├── CosmicStarfield.ts            # 5,500 Harvard spectral stars & nebulae
│   │   │   ├── GlowingTrail.ts               # Vertex-alpha orbital trajectory line shader
│   │   │   └── SunShader.ts                  # 3D Simplex solar granulation & corona shaders
│   │   ├── App.tsx                           # Application layout and state management
│   │   ├── index.css                         # Tailwind CSS v4 base styles
│   │   ├── main.tsx                          # React entry point
│   │   ├── physics.ts                        # Client integrators, presets, and metrics
│   │   └── types.ts                          # TypeScript interfaces
│   ├── index.html                            # HTML shell
│   ├── package.json                          # Frontend scripts and dependencies
│   ├── package-lock.json                     # Locked dependency tree
│   ├── tsconfig.json                         # TypeScript project references
│   ├── tsconfig.app.json                     # App TypeScript compiler config
│   ├── tsconfig.node.json                    # Vite config TypeScript settings
│   └── vite.config.ts                        # Vite configuration (port 5180)
├── docs/
│   ├── README.md                             # Documentation index
│   ├── wiki/
│   │   ├── README.md                         # Technical wiki index
│   │   ├── ARCHITECTURE.md                   # System architecture (English)
│   │   ├── ARCHITECTURE.es.md                # System architecture (Spanish)
│   │   ├── API_REFERENCE.md                  # API specification (English)
│   │   └── API_REFERENCE.es.md               # API specification (Spanish)
│   └── notes/
│       ├── README.md                         # Theoretical notes index
│       ├── 01_orbital_mechanics_guide.md     # Celestial mechanics & integrators (ES)
│       ├── 02_pinns_and_hnn_guide.md         # Symplectic PINNs & Liouville proof (ES)
│       ├── 03_hpc_architecture_guide.md      # C++20 HPC & OpenMP optimization (ES)
│       └── 04_webgl_shaders_guide.md         # GLSL shaders & ACES HDR pipeline (ES)
├── .gitignore
├── LICENSE                                   # Apache License 2.0
├── README.md                                 # English documentation
└── README.es.md                              # Spanish documentation
```

---

## License and Citation

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for full terms.

```bibtex
@software{amundarain2026astrodynamics3d,
  author = {Amundarain, Mois{\'e}s},
  title = {{AstroDynamics 3D: N-Body Gravitational Engine \& Symplectic PINN Surrogate}},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/moises-inc/astrodynamics-3d}
}
```
