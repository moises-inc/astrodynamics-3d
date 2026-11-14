# Nota Técnica: Mecánica Celeste e Integradores Gravitacionales N-Cuerpos

Este documento expone los principios físicos y matemáticos de las simulaciones gravitacionales de $N$ cuerpos en **AstroDynamics 3D** y los esquemas de integración numérica empleados para su propagación temporal.

---

## 1. Ley de Gravitación Universal de Newton

En un sistema gravitacional aislado de $N$ cuerpos celestes (estrellas, planetas, satélites), la fuerza de atracción gravitacional $\vec{F}_{ij}$ ejercida sobre la masa $m_i$ (en posición $\vec{r}_i$) por la masa $m_j$ (en posición $\vec{r}_j$) está dada por:

$$
\vec{F}_{ij} = G \frac{m_i m_j}{\|\vec{r}_j - \vec{r}_i\|^3} (\vec{r}_j - \vec{r}_i) = -G \frac{m_i m_j}{\|\vec{r}_i - \vec{r}_j\|^3} (\vec{r}_i - \vec{r}_j)
$$

Donde:
- $G$ es la constante de gravitación universal ($G \approx 6.67430 \times 10^{-11}\text{ m}^3\text{kg}^{-1}\text{s}^{-2}$, o normalizada a unidades adimensionales $G = 1.0$).
- $m_i, m_j$ son las masas gravitatorias de los cuerpos $i$ y $j$.
- $\vec{r}_j - \vec{r}_i$ es el vector de desplazamiento dirigido desde la masa $i$ hacia la masa $j$.
- $\|\vec{r}_j - \vec{r}_i\|$ es la distancia euclidiana en $\mathbb{R}^3$.

Por la Segunda Ley de Newton ($\vec{F}_i = m_i \vec{a}_i$), la aceleración neta de la partícula $i$ debida a la atracción combinada de todas las demás masas $j \neq i$ es:

$$
\vec{a}_i = \frac{d^2 \vec{r}_i}{dt^2} = G \sum_{\substack{j=1 \\ j \ne i}}^N \frac{m_j (\vec{r}_j - \vec{r}_i)}{\left(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}
$$

### Parámetro de Suavizado de Plummer ($\epsilon$)

Cuando dos partículas experimentan un encuentro cercano ($\vec{r}_i \to \vec{r}_j$), la distancia mutua $\|\vec{r}_j - \vec{r}_i\| \to 0$ origina una singularidad en el potencial puntual clásico que produce aceleraciones divergentes.

Para regularizar numéricamente el denominador, se introduce la longitud de suavizado esférico de Plummer $\epsilon > 0$:

$$
r_{ij, \epsilon} = \sqrt{\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2}
$$

El potencial gravitacional regularizado de Plummer queda definido como:

$$
\Phi_\epsilon(r) = -\frac{G M}{\sqrt{r^2 + \epsilon^2}}
$$

---

## 2. Integradores Numéricos: RK4 Clásico vs. Störmer-Verlet Simpléctico

Para propagar el sistema dinámico en el tiempo con paso discreto $\Delta t$, **AstroDynamics 3D** proporciona esquemas de integración complementarios.

### A. Integrador Clásico Runge-Kutta de 4.º Orden (RK4)

RK4 evalúa cuatro pendientes vectoriales intermedias ($k_1, k_2, k_3, k_4$) dentro del intervalo $[t, t + \Delta t]$:

$$
\begin{aligned}
k_1 &= f(t, y_t) \\
k_2 &= f\left(t + \frac{\Delta t}{2}, y_t + \frac{\Delta t}{2} k_1\right) \\
k_3 &= f\left(t + \frac{\Delta t}{2}, y_t + \frac{\Delta t}{2} k_2\right) \\
k_4 &= f(t + \Delta t, y_t + \Delta t \, k_3) \\
y_{t+\Delta t} &= y_t + \frac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
\end{aligned}
$$

- **Error de truncamiento:** $\mathcal{O}(\Delta t^5)$ por paso (error global $\mathcal{O}(\Delta t^4)$).
- **Limitación en dinámica orbital:** RK4 no es simpléctico. No conserva el volumen del espacio de fases ni la 2-forma simpléctica $\omega = dq \wedge dp$, lo que induce una deriva secular de energía en horizontes de integración prolongados.

### B. Integrador Velocity Verlet Simpléctico (Störmer-Verlet)

El integrador de Verlet es un método geométrico basado en la descomposición simétrica de Strang que respeta la estructura hamiltoniana del espacio de fases (Teorema de Liouville). Su algoritmo de actualización es:

$$
\begin{aligned}
\vec{r}(t + \Delta t) &= \vec{r}(t) + \vec{v}(t)\,\Delta t + \frac{1}{2}\,\vec{a}(t)\,\Delta t^2 \\
\vec{a}(t + \Delta t) &= \vec{a}\big(\vec{r}(t + \Delta t)\big) \\
\vec{v}(t + \Delta t) &= \vec{v}(t) + \frac{1}{2}\left[\vec{a}(t) + \vec{a}(t + \Delta t)\right]\Delta t
\end{aligned}
$$

- **Conservación geométrica:** Ausencia de deriva secular de energía. Conserva un Hamiltoniano sombra $\tilde{H} = H + \mathcal{O}(\Delta t^2)$, manteniendo las fluctuaciones energéticas acotadas en el tiempo.
- **Eficiencia computacional:** Al reutilizar $\vec{a}(t + \Delta t)$ en el ciclo siguiente mediante el búfer `cur_acc_`, requiere una única evaluación de fuerza $\mathcal{O}(N^2)$ por paso en régimen estacionario.

---

## 3. Leyes de Conservación e Invariantes Físicos

En un sistema gravitacional aislado en $\mathbb{R}^3$, la dinámica conservativa exige la invariancia de las siguientes cantidades:

### 1. Energía Total del Hamiltoniano ($H$)

$$
H(q, p) = T(p) + V(q) = \sum_{i=1}^N \frac{1}{2} m_i \|\vec{v}_i\|^2 - \sum_{1 \le i < j \le N} \frac{G m_i m_j}{\sqrt{\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2}}
$$

El error relativo de energía se monitoriza como:

$$
\varepsilon_H(t) = \frac{|H(t) - H(0)|}{|H(0)|}
$$

### 2. Momento Angular Total ($\vec{L}$)

Por isotropía del espacio (invariancia bajo rotaciones en $SO(3)$ según el Teorema de Noether):

$$
\vec{L} = \sum_{i=1}^N m_i (\vec{r}_i \times \vec{v}_i) = \text{constante}
$$

### 3. Momento Lineal Total y Centro de Masas ($\vec{P}_{\text{tot}}, \vec{R}_{\text{cm}}$)

Por homogeneidad del espacio (invariancia bajo traslaciones):

$$
\vec{P}_{\text{tot}} = \sum_{i=1}^N m_i \vec{v}_i = \text{constante}, \qquad \vec{R}_{\text{cm}} = \frac{1}{M_{\text{tot}}} \sum_{i=1}^N m_i \vec{r}_i
$$

Donde $M_{\text{tot}} = \sum_{i=1}^N m_i$ es la masa total del sistema.

---

## 4. Definiciones Fundamentales

- **Problema de $N$ Cuerpos:** Determinación numérica de las trayectorias de $N$ masas puntuales que interactúan gravitacionalmente según las leyes de Newton.
- **Espacio de Fases:** Variedad de $6N$ dimensiones compuesta por las posiciones generalizadas $q = (\vec{r}_1, \dots, \vec{r}_N)$ y momentos conjugados $p = (\vec{p}_1, \dots, \vec{p}_N)$.
- **Integrador Simpléctico:** Esquema numérico que preserva exactamente la 2-forma simpléctica cerrada no degenerada $\omega = \sum dq_i \wedge dp_i$, acotando el error energético en simulaciones de larga duración.
