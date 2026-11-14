# Arquitectura del Sistema AstroDynamics 3D

Este documento especifica la arquitectura de software y el diseño matemático de **AstroDynamics 3D**, un simulador gravitacional híbrido de $N$ cuerpos que integra un núcleo numérico en C++20, una capa surrogate de Red Neuronal Informada por la Física (PINN) simpléctica en PyTorch y un cliente interactivo WebGL 3D.

---

## 1. Descomposición Modular del Sistema

AstroDynamics 3D está estructurado en tres capas desacopladas:

```text
+-------------------------------------------------------------------+
|                  Frontend WebGL 3D Interactivo                    |
|         (React 19 + TypeScript + Three.js + Tailwind v4)          |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                Capa Surrogate PINN Simpléctica (PyTorch)          |
|    Parametrización H_theta(q, p) | Campo Canónico Autodif         |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                  Núcleo C++20 de Alto Rendimiento                 |
|      Störmer-Verlet | OpenMP | SIMD AVX2/FMA sin bifurcación      |
+-------------------------------------------------------------------+
```

---

## 2. Capa 1: Núcleo Numérico C++20 (`cpp_core/`)

Archivos fuente: [`cpp_core/include/nbody_solver.hpp`](../../cpp_core/include/nbody_solver.hpp), [`cpp_core/src/nbody_solver.cpp`](../../cpp_core/src/nbody_solver.cpp).

### 2.1 Potencial Gravitacional con Suavizado de Plummer

La interacción gravitacional entre pares de partículas se regulariza mediante el potencial de esfera de Plummer con escala $\epsilon > 0$:

$$
\Phi(r) = -\frac{G M}{\sqrt{r^2 + \epsilon^2}}
$$

La fuerza ejercida sobre el cuerpo $i$ por el cuerpo $j$ es:

$$
\vec{F}_{ij} = \frac{G m_i m_j (\vec{r}_j - \vec{r}_i)}{\left(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}
$$

### 2.2 Vectorización SIMD sin Bifurcaciones y Layout SoA

En implementaciones convencionales, una condición explícita (`if (i == j) continue;`) omite la auto-interacción, interrumpiendo el flujo en registros vectoriales SIMD. Cuando $\epsilon > 0$, la auto-interacción produce:

$$
\vec{r}_i - \vec{r}_i = \mathbf{0} \implies \vec{a}_{ii} = \frac{G m_i \mathbf{0}}{\left(0 + \epsilon^2\right)^{3/2}} = \mathbf{0}
$$

Al suprimir la bifurcación, el núcleo emplea `#pragma omp simd reduction(+:ax, ay, az)` y auto-vectorización FMA sobre la caché contigua de masas `masses_` (layout *Structure of Arrays*) y las vistas `std::span` de posiciones. Cuando `softening == 0.0`, el cálculo conmuta a una ruta protegida para evitar divisiones por cero.

### 2.3 Verlet Simpléctico con Caché de Aceleraciones

Una implementación directa de Velocity Verlet requiere dos evaluaciones $\mathcal{O}(N^2)$ de fuerza por paso temporal:

1. $\vec{r}(t + \Delta t) = \vec{r}(t) + \vec{v}(t)\Delta t + \frac{1}{2}\vec{a}(t)\Delta t^2$
2. Evaluar $\vec{a}(t + \Delta t)$
3. $\vec{v}(t + \Delta t) = \vec{v}(t) + \frac{1}{2}[\vec{a}(t) + \vec{a}(t + \Delta t)]\Delta t$

`NBodySystem` mantiene búferes persistentes preasignados (`cur_acc_`, `scratch_pos_`, `scratch_acc_`, y los búferes de etapas RK4 `r0_`, `v0_`, `r_scratch_`, `k1_v_`–`k4_v_`). La aceleración $\vec{a}(t+\Delta t)$ calculada en el paso (2) se intercambia hacia `cur_acc_` para el ciclo siguiente, reduciendo a la mitad las evaluaciones de fuerza en régimen estacionario ($1 \times \mathcal{O}(N^2)$ por paso) y logrando cero asignaciones en *heap* por paso tras la inicialización.

### 2.4 Invariantes Físicos y Paso Temporal Adaptativo

- **Momento Lineal y Control Baricéntrico:** `reset_barycenter()` impone momento lineal total nulo $\vec{P}_{\text{tot}} = \sum_{i=1}^N m_i \vec{v}_i = \mathbf{0}$ y opcionalmente centra $\vec{R}_{\text{cm}}$.
- **Hamiltoniano Sombra:** Por la expansión de Baker-Campbell-Hausdorff (BCH), la descomposición de Strang preserva un Hamiltoniano modificado $\tilde{H} = H + \Delta t^2 H_2 + \mathcal{O}(\Delta t^4)$, acotando las oscilaciones de energía sin deriva secular.
- **Sub-paso Adaptativo:** `compute_adaptive_dt()` ajusta el paso respecto a la mínima escala temporal dinámica entre pares $\tau_{ij} = \sqrt{r_{ij}^3 / (G (m_i + m_j))}$.

---

## 3. Capa 2: Capa Surrogate PINN Simpléctica (`pinn_surrogate/`)

Archivo fuente: [`pinn_surrogate/model.py`](../../pinn_surrogate/model.py).

### 3.1 Parametrización mediante Red Neuronal Hamiltoniana

En lugar de predecir aceleraciones directamente (lo cual introduce disipación o ganancia artificial de energía), `HamiltonianNN` parametriza la función escalar del Hamiltoniano $H_\theta(q, p): \mathbb{R}^{2d} \to \mathbb{R}$. Cuando `separable=True`, descompone la energía en ramas cinética y potencial:

$$
H_\theta(q, p) = T_\theta(p) + V_\theta(q)
$$

Las derivadas canónicas en el espacio de fases se extraen mediante diferenciación automática en PyTorch:

$$
\hat{\dot{q}} = \frac{\partial H_\theta}{\partial p}, \qquad \hat{\dot{p}} = -\frac{\partial H_\theta}{\partial q}
$$

### 3.2 Formulación de la Función de Pérdida Simpléctica

`SymplecticPINNLoss` impone las ecuaciones canónicas de Hamilton y opcionalmente ancla la referencia de energía respecto a $H_0$:

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{k=1}^B \left( \left\|\frac{\partial H_\theta}{\partial p} - \dot{q}_k\right\|_2^2 + \left\|\frac{\partial H_\theta}{\partial q} + \dot{p}_k\right\|_2^2 \right) + \lambda_E \frac{1}{B}\sum_{k=1}^B \|H_\theta(q_k, p_k) - H_0\|^2
$$

El uso de funciones de activación suaves $C^\infty$ (`Tanh` o `SiLU`) asegura derivadas segundas mixtas bien definidas durante el entrenamiento.

### 3.3 Integración con Euler Simpléctico

`symplectic_euler_step()` y `rollout_surrogate()` propagan trayectorias mediante el esquema semi-implícito de Euler simpléctico:

$$
\begin{aligned}
p_{n+1} &= p_n - \Delta t \left.\frac{\partial H_\theta}{\partial q}\right|_{(q_n, p_n)} \\
q_{n+1} &= q_n + \Delta t \left.\frac{\partial H_\theta}{\partial p}\right|_{(q_n, p_{n+1})}
\end{aligned}
$$

Para Hamiltonianos separables, el determinante Jacobiano cumple $\det J = 1$, conservando exactamente la 2-forma simpléctica $\omega = dq \wedge dp$ y el volumen de fase de Liouville.

---

## 4. Capa 3: Frontend WebGL 3D Interactivo (`web_ui/`)

Directorio fuente: [`web_ui/src/`](../../web_ui/src).

### 4.1 Pipeline de Renderizado y Grafo de Escena

- **Tecnología:** React 19, TypeScript, Vite 6, Tailwind CSS v4 y Three.js.
- **Post-Procesamiento:** `EffectComposer` encadenando `RenderPass` y `UnrealBloomPass` con mapeo de tonos `ACESFilmicToneMapping` (exposición `1.1`, umbral `0.18`).
- **Elementos de Escena:**
  - **Campo Estelar Cósmico** ([`CosmicStarfield.ts`](../../web_ui/src/visuals/CosmicStarfield.ts)): 5,500 partículas estelares muestreadas según la secuencia espectral de Harvard (clases O a M) con sombreadores de disco gaussiano y 6 planos volumétricos de nebulosa.
  - **Shaders Solares y Atmosféricos** ([`SunShader.ts`](../../web_ui/src/visuals/SunShader.ts), [`AtmosphereShader.ts`](../../web_ui/src/visuals/AtmosphereShader.ts)): granulación fotosférica con ruido Simplex 3D y pulsación coronal para estrellas emisivas; dispersión atmosférica de Rayleigh mediante Fresnel invertido para planetas.
  - **Estelas Orbitales** ([`GlowingTrail.ts`](../../web_ui/src/visuals/GlowingTrail.ts)): geometría dinámica `THREE.Line` con atenuación alfa por vértice desde la cabeza hacia la cola.
  - **Rejilla de Referencia:** `GridHelper` de 50x50 divisiones en el plano eclíptico.
- **Ciclo de Vida GPU:** `disposeHierarchy()` libera de forma explícita geometrías, materiales, pases de post-procesamiento y contexto WebGL al desmontar el componente, verificado sin fugas de memoria mediante [`verify_webgl_disposal.cjs`](../../web_ui/scripts/verify_webgl_disposal.cjs).

### 4.2 Controles de Simulación y Telemetría en Tiempo Real

- **Motor de Física en Cliente** ([`physics.ts`](../../web_ui/src/physics.ts)): implementa `stepSymplecticVerlet`, `stepRK4`, `stepPINNSurrogate`, paso temporal adaptativo (`computeAdaptiveTimeStep`), corrección baricéntrica (`resetBarycentricDrift`) y configuraciones predefinidas (`kepler`, `figure8`, `lagrange`).
- **Panel de Control** ([`ControlsPanel.tsx`](../../web_ui/src/components/ControlsPanel.tsx)): permite ajustar $G$, $\Delta t$, esquema de integración, longitud de estela, bloom HDR, masas y velocidades por cuerpo, y visualiza telemetría de invariantes (~15 Hz: $H$, $\Delta H / |H_0|$, $\|\vec{V}_{\text{cm}}\|$, $\|\vec{L}\|$, $r_{\min}$) con gráfico logarítmico en tiempo real.
