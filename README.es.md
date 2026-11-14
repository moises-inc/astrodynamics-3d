# AstroDynamics 3D: Motor Gravitacional N-Body y Surrogate PINN Simpléctico

[![Estado: v1.0.0 - Archived](https://img.shields.io/badge/Estado-v1.0.0_--_Archived-lightgrey.svg)](#)
[![Idioma: Español](https://img.shields.io/badge/Idioma-Espa%C3%B1ol-green.svg)](#)
[![Version: English](https://img.shields.io/badge/Version-English-blue.svg)](README.md)
[![Licencia: Apache-2.0](https://img.shields.io/badge/Licencia-Apache_2.0-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19.0+-61dafb.svg)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-r173+-black.svg)](https://threejs.org/)

> **Motor numérico de mecánica celeste de alto rendimiento con Redes Neuronales Informadas por la Física (PINNs simplécticas) y visualización interactiva WebGL 3D.**
>
> Autor: **Moisés Amundarain** | Read in English: [README.md](README.md)

> **Proyecto Archivado (`v1.0.0 - Archived`):** Este repositorio se encuentra completado y archivado como referencia.

---

## Fundamentos Matemáticos y Físicos

### 1. Mecánica Celeste N-Body con Suavizado de Plummer

Las interacciones gravitacionales entre pares de cuerpos se regularizan mediante el potencial de la esfera de Plummer con escala $\epsilon > 0$ para evitar singularidades en encuentros cercanos:

$$
U(r_1, \dots, r_N) = -\sum_{1 \le i < j \le N} \frac{G m_i m_j}{\sqrt{\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2}}
$$

$$
\vec{a}_i = -\frac{1}{m_i} \nabla_{\vec{r}_i} U = \sum_{j \neq i} \frac{G m_j (\vec{r}_j - \vec{r}_i)}{\left(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}
$$

**Evaluación SIMD sin Bifurcaciones:** Para la auto-interacción ($j = i$), $(\vec{r}_i - \vec{r}_i) = \mathbf{0}$, por lo que el término de fuerza evalúa idénticamente a $\mathbf{0}$ cuando $\epsilon > 0$. Esto elimina la bifurcación condicional dentro del bucle interno, habilitando vectorización continua AVX2/FMA y reducciones SIMD en OpenMP.

### 2. Geometría Simpléctica y Conservación de Invariantes

El espacio de fases $\mathcal{M} = \mathbb{R}^{2d}$ está dotado de la 2-forma simpléctica canónica:

$$
\omega = \sum_{k=1}^d dq_k \wedge dp_k
$$

- **Simplecticidad Exacta:** El flujo discreto $\Phi_{\Delta t}$ preserva la 2-forma diferencial: $\Phi_{\Delta t}^* \omega = \omega$.
- **Volumen de Liouville:** La conservación de $\Omega = \bigwedge^d \omega$ implica $\det\left(\frac{\partial(q_{n+1}, p_{n+1})}{\partial(q_n, p_n)}\right) = 1$.
- **Hamiltoniano Sombra (Expansión BCH):** El esquema Velocity Verlet simpléctico preserva un Hamiltoniano modificado $\tilde{H}(q, p) = H(q, p) + \Delta t^2 H_2(q, p) + \mathcal{O}(\Delta t^4)$, acotando las oscilaciones de energía sin deriva secular en integraciones prolongadas.

### 3. Redes Neuronales Hamiltonianas (HNN) y PINNs Simplécticas

En lugar de predecir aceleraciones directamente, la capa surrogate en PyTorch parametriza el escalar de energía Hamiltoniana $H_\theta(q, p)$:

$$
\dot{q} = \frac{\partial H_\theta}{\partial p}, \qquad \dot{p} = -\frac{\partial H_\theta}{\partial q}
$$

La función de pérdida simpléctica impone las ecuaciones canónicas de Hamilton mediante diferenciación automática junto con un término de anclaje energético:

$$
\mathcal{L}_{\text{total}}(\theta) = \left\| \frac{\partial H_\theta}{\partial p} - \dot{q}_{\text{true}} \right\|^2 + \left\| \frac{\partial H_\theta}{\partial q} + \dot{p}_{\text{true}} \right\|^2 + \lambda_E \|H_\theta(q_t, p_t) - H_0\|^2
$$

---

## Documentación

Las especificaciones técnicas y las notas teóricas están organizadas en [`docs/`](docs/README.md):

| Categoría | Documento | Descripción |
| :--- | :--- | :--- |
| **Índice** | [docs/README.md](docs/README.md) | Índice maestro de la documentación |
| **Wiki (Referencia)** | [docs/wiki/ARCHITECTURE.es.md](docs/wiki/ARCHITECTURE.es.md) ([EN](docs/wiki/ARCHITECTURE.md)) | Arquitectura en tres capas, integradores, capa PINN y pipeline WebGL |
| **Wiki (Referencia)** | [docs/wiki/API_REFERENCE.es.md](docs/wiki/API_REFERENCE.es.md) ([EN](docs/wiki/API_REFERENCE.md)) | Referencia de APIs públicas en `cpp_core`, `pinn_surrogate` y `web_ui` |
| **Notas (Teoría)** | [docs/notes/01_orbital_mechanics_guide.md](docs/notes/01_orbital_mechanics_guide.md) | Dinámica gravitacional N-cuerpos, Plummer y RK4 vs. Störmer-Verlet |
| **Notas (Teoría)** | [docs/notes/02_pinns_and_hnn_guide.md](docs/notes/02_pinns_and_hnn_guide.md) | Formulación Hamiltoniana, pérdida simpléctica y demostración de Liouville |
| **Notas (Teoría)** | [docs/notes/03_hpc_architecture_guide.md](docs/notes/03_hpc_architecture_guide.md) | Layout SoA, búferes *zero-allocation* y escalamiento con OpenMP |
| **Notas (Teoría)** | [docs/notes/04_webgl_shaders_guide.md](docs/notes/04_webgl_shaders_guide.md) | Shaders de Rayleigh, ruido Simplex solar, campo estelar y HDR ACES |

---

## Verificación y Suites de Prueba

| Componente | Suite / Script | Cobertura Verificada |
| :--- | :--- | :--- |
| **Núcleo C++20** | GoogleTest / CTest (`test_nbody`) | 10 pruebas: centro de masa, conservación energética en Verlet y RK4, momento angular, paso adaptativo, reinicio baricéntrico y suavizado nulo |
| **Memoria y Rendimiento C++20** | Benchmark (`benchmark_hpc`) | `0` asignaciones en *heap* por paso en régimen estacionario para $N \in \{2, 3, 128, 512, 1024\}$ en Verlet y RK4; escalamiento multihilo OpenMP |
| **PINN Simpléctica** | Pytest (`test_pinn_conservation.py`) | 4 pruebas: verificación de gradientes autodiff frente a diferencias finitas, retropropagación simpléctica, estabilidad energética a 1,000 pasos y HNN separable |
| **Frontend WebGL y GPU** | TypeScript + Vite + Node (`verify_webgl_disposal.cjs`) | Compilación limpia `tsc -b && vite build`; auditoría de liberación de geometrías/materiales en 500 ciclos de montaje/desmontaje |
| **E2E en Navegador (Opcional)** | Playwright (`e2e_browser_test_astrodynamics.cjs`) | Cambio de presets, navegación por pestañas `Invariants` / `Bodies` y verificación sin errores de página en `http://localhost:5180` |

---

## Guía de Inicio Rápido

### Requisitos Previos

- **Compilador C++:** GCC 13+ o Clang 17+ con soporte C++20 y OpenMP (`libomp-dev` / `libgomp`).
- **Sistema de Compilación:** CMake 3.20+.
- **Python:** Python 3.10+ (recomendado `3.11` o `3.12`) con `venv` / `pip`.
- **Node.js:** Node.js 18+ o 20+ con `npm`.

### 1. Compilación y Pruebas del Núcleo C++20

```bash
cmake -B cpp_core/build -S cpp_core -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_core/build -j$(nproc)
ctest --test-dir cpp_core/build --output-on-failure

# Benchmark HPC y auditoría de cero asignaciones dinámicas
./cpp_core/build/benchmark_hpc
```

### 2. Ejecución del Surrogate PINN Simpléctico

```bash
cd pinn_surrogate
python3 -m venv .venv
source .venv/bin/activate

# Instalar PyTorch (CPU), NumPy y pytest según pyproject.toml
pip install -e ".[test]"

# Ejecutar pruebas unitarias y de conservación
pytest tests/ -v

# Ejecutar demostración de inferencia y benchmark
python3 demo_inference.py
python3 benchmark_pinn.py
```

### 3. Ejecución del Frontend WebGL 3D

```bash
cd web_ui
npm install

# Iniciar servidor de desarrollo Vite en el puerto 5180
npm run dev

# O compilar y previsualizar el bundle de producción
npm run build
npm run preview
```

Abra `http://localhost:5180` en un navegador compatible con WebGL.

**Controles Interactivos:**
- **Clic izquierdo + arrastre:** Rotación orbital de cámara 3D.
- **Clic derecho + arrastre:** Paneo de cámara.
- **Rueda del ratón:** Acercar / alejar.
- **AstroDynamics Mission Deck:**
  - **Physics:** Selección de presets (`kepler`, `figure8`, `lagrange`), activación de paso temporal adaptativo, bloqueo baricéntrico y bloom HDR, ajuste de $G$ y $\Delta t$, y elección entre `Symplectic Verlet`, `Runge-Kutta 4th Order` y `Symplectic PINN Surrogate`.
  - **Bodies:** Edición en vivo de masa $m$ y vector de velocidad $(v_x, v_y, v_z)$ por cuerpo.
  - **Invariants:** Monitoreo de $H$, $H_0$, $\Delta H / |H_0|$ con gráfico logarítmico, $\|\vec{V}_{\text{cm}}\|$, $\|\vec{L}\|$, $r_{\min}$ y $\Delta t$ efectivo.

**Auditorías WebGL Opcionales:**

```bash
# Prueba de estrés de liberación de recursos GPU (500 ciclos)
node --expose-gc web_ui/scripts/verify_webgl_disposal.cjs

# Prueba E2E con Playwright (requiere servidor activo en el puerto 5180)
node web_ui/scripts/e2e_browser_test_astrodynamics.cjs
```

---

## Estructura del Repositorio

```text
astrodynamics-3d/
├── cpp_core/
│   ├── include/
│   │   └── nbody_solver.hpp                  # NBodySystem, Vec3, Body, IntegratorType
│   ├── src/
│   │   └── nbody_solver.cpp                  # Caché SoA, OpenMP SIMD, Verlet, RK4, dt adaptativo
│   ├── tests/
│   │   ├── test_nbody.cpp                    # Suite GoogleTest de invariantes físicos (10 pruebas)
│   │   └── benchmark_hpc.cpp                 # Benchmark de escalamiento y cero asignaciones
│   └── CMakeLists.txt                        # C++20, -O3, -march=native, OpenMP, GoogleTest
├── pinn_surrogate/
│   ├── tests/
│   │   └── test_pinn_conservation.py         # Pruebas pytest de autodiff y conservación a 1,000 pasos
│   ├── benchmark_pinn.py                     # Benchmark de latencia y memoria en rollout HNN
│   ├── demo_inference.py                     # Demo CLI de integración Hamiltoniana
│   ├── model.py                              # HamiltonianNN, SymplecticPINNLoss, Euler rollout
│   └── pyproject.toml                        # Metadatos y dependencias del paquete Python
├── web_ui/
│   ├── scripts/
│   │   ├── e2e_browser_test_astrodynamics.cjs # Verificación E2E automatizada con Playwright
│   │   └── verify_webgl_disposal.cjs         # Prueba de liberación de recursos WebGL
│   ├── src/
│   │   ├── components/
│   │   │   ├── ControlsPanel.tsx             # Panel de control y HUD de invariantes
│   │   │   └── OrbitCanvas3D.tsx             # Escena Three.js, EffectComposer, bloom y bucle
│   │   ├── visuals/
│   │   │   ├── AtmosphereShader.ts           # Halo atmosférico de Rayleigh (Fresnel invertido)
│   │   │   ├── CosmicStarfield.ts            # 5,500 estrellas espectrales de Harvard y nebulosas
│   │   │   ├── GlowingTrail.ts               # Estela orbital con gradiente alfa por vértice
│   │   │   └── SunShader.ts                  # Shaders solares de granulación Simplex y corona
│   │   ├── App.tsx                           # Contenedor principal y estado de simulación
│   │   ├── index.css                         # Estilos base Tailwind CSS v4
│   │   ├── main.tsx                          # Punto de entrada React
│   │   ├── physics.ts                        # Integradores en cliente, presets y métricas
│   │   └── types.ts                          # Definiciones de tipos TypeScript
│   ├── index.html                            # Documento HTML principal
│   ├── package.json                          # Scripts y dependencias del frontend
│   ├── package-lock.json                     # Árbol bloqueado de dependencias
│   ├── tsconfig.json                         # Referencias de proyecto TypeScript
│   ├── tsconfig.app.json                     # Configuración TypeScript de la aplicación
│   ├── tsconfig.node.json                    # Configuración TypeScript de Vite
│   └── vite.config.ts                        # Configuración Vite (puerto 5180)
├── docs/
│   ├── README.md                             # Índice maestro de documentación
│   ├── wiki/
│   │   ├── README.md                         # Índice de wiki técnica
│   │   ├── ARCHITECTURE.md                   # Arquitectura del sistema (Inglés)
│   │   ├── ARCHITECTURE.es.md                # Arquitectura del sistema (Español)
│   │   ├── API_REFERENCE.md                  # Especificación de API (Inglés)
│   │   └── API_REFERENCE.es.md               # Especificación de API (Español)
│   └── notes/
│       ├── README.md                         # Índice de notas teóricas
│       ├── 01_orbital_mechanics_guide.md     # Mecánica celeste e integradores
│       ├── 02_pinns_and_hnn_guide.md         # PINNs simplécticas y teorema de Liouville
│       ├── 03_hpc_architecture_guide.md      # Arquitectura HPC en C++20 y OpenMP
│       └── 04_webgl_shaders_guide.md         # Shaders GLSL y pipeline HDR ACES
├── .gitignore
├── LICENSE                                   # Licencia Apache 2.0
├── README.md                                 # Documentación en Inglés
└── README.es.md                              # Documentación en Español
```

---

## Licencia y Cita Académica

Distribuido bajo la **Licencia Apache 2.0**. Consulte [`LICENSE`](LICENSE) para conocer los términos completos.

```bibtex
@software{amundarain2026astrodynamics3d,
  author = {Amundarain, Mois{\'e}s},
  title = {{AstroDynamics 3D: N-Body Gravitational Engine \& Symplectic PINN Surrogate}},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/moises-inc/astrodynamics-3d}
}
```
