# Nota Técnica: Arquitectura C++20 HPC y Optimización de Rendimiento

Este documento describe los principios de optimización computacional aplicados en el núcleo numérico C++20 de **AstroDynamics 3D** ([`cpp_core/`](../../cpp_core)), centrados en la localidad espacial en caché L1, la vectorización SIMD sin bifurcaciones, el manejo de memoria libre de asignaciones dinámicas y la paralelización con OpenMP.

---

## 1. Estructura de Datos: AoS vs. SoA (*Structure of Arrays*)

En simulaciones gravitacionales de $N$ cuerpos, el bucle interno de aceleración de complejidad $\mathcal{O}(N^2)$ requiere acceder repetidamente a las masas $m_j$ y posiciones $\vec{r}_j$ de todas las partículas.

### A. Layout Convencional (*Array of Structures*, AoS)

```cpp
struct Body {
    std::string name; // ~32 bytes
    double mass;      // 8 bytes
    Vec3 position;    // 24 bytes
    Vec3 velocity;    // 24 bytes
};
std::vector<Body> bodies;
```

En una línea de caché L1 estándar de 64 bytes, cargar una estructura `Body` completa (~88–96 bytes) arrastra campos que no intervienen en el cálculo de fuerzas (`name`, `velocity`), reduciendo la densidad útil de transferencia desde memoria principal.

### B. Caché Contigua de Masas (*Structure of Arrays*, SoA)

En [`NBodySystem`](../../cpp_core/include/nbody_solver.hpp), las masas se sincronizan en un arreglo contiguo dedicado:

```cpp
mutable std::vector<double> masses_; // 8 bytes * N contiguos
```

Una línea de caché L1 de 64 bytes transfiere 8 escalares `double` consecutivos, maximizando la localidad espacial y facilitando la carga hacia registros vectoriales SIMD.

---

## 2. Complejidad Computacional y FLOPs

Para $N$ partículas, la evaluación directa de interacciones entre todos los pares ordenados procesa $N^2$ términos (o $N(N-1)/2$ pares simétricos en la evaluación escalar de energía potencial). En el kernel suavizado de Plummer:

$$
\vec{a}_{ij} = \frac{G m_j (\vec{r}_j - \vec{r}_i)}{\left(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}
$$

cada par $(i, j)$ requiere del orden de $20\text{ FLOPs}$ (3 restas, 3 productos y 2 sumas para la norma al cuadrado más $\epsilon^2$, 1 raíz cuadrada e inversa, 2 productos para el cubo y escalado, y 6 operaciones de multiplicación-acumulación FMA). En el bucle completo $\mathcal{O}(N^2)$ de aceleración:

$$
\text{FLOPs/paso} \approx 20 N^2
$$

Para $N = 1024$, cada evaluación de aceleraciones ejecuta aproximadamente $2.1 \times 10^7\text{ FLOPs}$.

---

## 3. Integración sin Asignaciones en Heap (*Zero-Allocation*)

Invocar asignadores dinámicos (`malloc`, `new`, `std::vector::resize`) dentro del ciclo de integración introduce latencia variable y contención en entornos multihilo.

`NBodySystem` preasigna todos los búferes de trabajo durante la inicialización o cambio de tamaño del sistema:
- **Velocity Verlet:** `cur_acc_`, `scratch_pos_`, `scratch_acc_`.
- **RK4:** `r0_`, `v0_`, `r_scratch_`, `k1_v_`, `k2_r_`, `k2_v_`, `k3_r_`, `k3_v_`, `k4_r_`, `k4_v_`.

Mediante la reutilización de `cur_acc_` con `std::swap(cur_acc_, scratch_acc_)`, cada paso de Velocity Verlet ejecuta una sola evaluación de fuerza $\mathcal{O}(N^2)$ y mantiene cero asignaciones dinámicas en régimen estacionario.

---

## 4. Paralelización Multihilo con OpenMP y Resultados Empíricos

El bucle externo sobre las partículas se paraleliza con OpenMP cuando $N \ge 128$ para evitar sobrecarga de sincronización en sistemas pequeños:

```cpp
#pragma omp parallel for schedule(static) if(n >= 128)
for (long long i = 0; i < static_cast<long long>(n); ++i) {
    // Cálculo vectorial branchless de aceleraciones
}
```

El escalamiento paralelo sigue la Ley de Amdahl:

$$
S(p) = \frac{T_1}{T_p}
$$

donde $T_1$ es el tiempo de ejecución monohilo y $T_p$ el tiempo con $p$ hilos.

### Mediciones de Referencia (`benchmark_hpc`)

Resultados observados al ejecutar [`cpp_core/tests/benchmark_hpc.cpp`](../../cpp_core/tests/benchmark_hpc.cpp) sobre 1,000 pasos en compilación `Release` (`-O3 -march=native`):

| $N$ | Método | Hilos | Tiempo Total (ms) | Rendimiento (pasos/s) | Asignaciones Steady-State | Speedup vs. 1 hilo |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| 512 | Verlet | 1 | 580.78 | 1,721.8 | 0 | 1.00x |
| 512 | Verlet | 4 | 287.98 | 3,472.5 | 0 | 2.02x |
| 1024 | Verlet | 1 | 2,523.84 | 396.2 | 0 | 1.00x |
| 1024 | Verlet | 4 | 1,254.40 | 797.2 | 0 | 2.01x |
| 1024 | RK4 | 1 | 7,617.02 | 131.3 | 0 | 1.00x |
| 1024 | RK4 | 4 | 3,685.89 | 271.3 | 0 | 2.07x |

La auditoría de memoria con sobrecarga global de `operator new` confirma `0` asignaciones en *heap* por paso en régimen estacionario tanto para Verlet como para RK4 en todos los tamaños evaluados ($N \in \{2, 3, 128, 512, 1024\}$).
