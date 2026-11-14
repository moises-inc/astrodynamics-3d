# Nota Técnica: Redes Neuronales Informadas por la Física (PINNs) y HNNs Simplécticas

Este documento detalla la formulación matemática de la capa de aprendizaje automático de **AstroDynamics 3D**, explicando cómo las Redes Neuronales Hamiltonianas (HNN) y las PINNs aproximan la dinámica de sistemas conservativos preservando los invariantes canónicos.

---

## 1. Principio de las Redes Neuronales Informadas por la Física (PINN)

Una **PINN** (*Physics-Informed Neural Network*) incorpora directamente las ecuaciones diferenciales y leyes de conservación en su función de pérdida mediante diferenciación automática analítica:

```text
[Red Neuronal Estándar]   ---> Minimiza únicamente el error de ajuste a datos
[PINN / HNN Simpléctica]  ---> Error de ajuste + residuos canónicos y energéticos (L_symplectic + L_energy)
```

---

## 2. Formulación Hamiltoniana en el Espacio de Fases: $H_{\theta}(q, p)$

En mecánica analítica, el estado instantáneo de un sistema de $N$ partículas en $\mathbb{R}^3$ se representa mediante el vector de estado $\mathbf{z} \in \mathbb{R}^{6N}$:

$$
\mathbf{z} = \begin{pmatrix} q \\ p \end{pmatrix} = \begin{pmatrix} \vec{r}_1 \\ \vdots \\ \vec{r}_N \\ \vec{p}_1 \\ \vdots \\ \vec{p}_N \end{pmatrix} \in \mathbb{R}^{6N}
$$

Donde:
- $q = (\vec{r}_1^T, \dots, \vec{r}_N^T)^T \in \mathbb{R}^{3N}$ representa las posiciones generalizadas cartesianas.
- $p = (\vec{p}_1^T, \dots, \vec{p}_N^T)^T \in \mathbb{R}^{3N}$ representa los momentos lineales conjugados ($\vec{p}_i = m_i \vec{v}_i \in \mathbb{R}^3$).

El **Hamiltoniano** $H(q, p)$ expresa la energía total como suma de la energía cinética $T(p)$ y la energía potencial gravitacional $U(q)$ (denotada también como $V(q)$):

$$
H(q, p) = T(p) + U(q) = \frac{1}{2} p^T M^{-1} p - \sum_{1 \le i < j \le N} \frac{G m_i m_j}{\sqrt{\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2}}
$$

Donde:
- $T(p) = \frac{1}{2} p^T M^{-1} p = \sum_{i=1}^N \frac{\|\vec{p}_i\|^2}{2 m_i}$.
- $U(q) = -\sum_{1 \le i < j \le N} \frac{G m_i m_j}{\sqrt{\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2}}$ con parámetro de suavizado $\epsilon > 0$.
- $M = \text{diag}(m_1 I_3, \dots, m_N I_3) \in \mathbb{R}^{3N \times 3N}$ es la matriz diagonal invertible de masas.

Las **Ecuaciones Canónicas de Hamilton** determinan la evolución temporal del flujo dinámico:

$$
\dot{q} = \nabla_p H(q, p) = M^{-1} p, \qquad \dot{p} = -\nabla_q H(q, p) = -\nabla_q U(q)
$$

Para cada cuerpo $i \in \{1, \dots, N\}$:

$$
\dot{\vec{r}}_i = \frac{\vec{p}_i}{m_i}, \qquad \dot{\vec{p}}_i = \vec{F}_i = -\nabla_{\vec{r}_i} U(q) = \sum_{\substack{j=1 \\ j \ne i}}^N \frac{G m_i m_j (\vec{r}_j - \vec{r}_i)}{\left(\|\vec{r}_j - \vec{r}_i\|^2 + \epsilon^2\right)^{3/2}}
$$

En notación simpléctica matricial:

$$
\dot{\mathbf{z}} = \Omega \nabla_{\mathbf{z}} H(\mathbf{z}), \qquad \Omega = \begin{pmatrix} 0 & I_{3N} \\ -I_{3N} & 0 \end{pmatrix} \in \mathbb{R}^{6N \times 6N}
$$

Dado que $\Omega^T = -\Omega$ y $\Omega^2 = -I_{6N}$, la matriz canónica $\Omega$ es ortogonal y antisimétrica ($\Omega^{-1} = -\Omega = \Omega^T$).

### Parametrización Separable mediante Autodiferenciación

**AstroDynamics 3D** parametriza la energía escalar mediante una red $H_{\theta}(q, p): \mathbb{R}^{2d} \to \mathbb{R}$ con pesos $\theta$. En la variante separable (`separable=True` en [`pinn_surrogate/model.py`](../../pinn_surrogate/model.py)):

$$
H_{\theta}(q, p) = T_{\theta}(p) + V_{\theta}(q)
$$

donde `net_T` aproxima la rama cinética $T(p)$ y `net_V` aproxima la rama potencial $V(q)$. Las derivadas temporales canónicas se calculan mediante `torch.autograd.grad`:

$$
\hat{\dot{q}} = \nabla_p H_{\theta}(q, p) = \frac{\partial H_{\theta}}{\partial p}, \qquad \hat{\dot{p}} = -\nabla_q H_{\theta}(q, p) = -\frac{\partial H_{\theta}}{\partial q}
$$

---

## 3. Función de Pérdida Simpléctica Compuesta

Para ajustar el campo vectorial hamiltoniano sin disipación espuria, el módulo [`pinn_surrogate/model.py`](../../pinn_surrogate/model.py) define `SymplecticPINNLoss`:

$$
\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{symplectic}}(\theta) + \lambda_E \mathcal{L}_{\text{energy}}(\theta)
$$

### A. Pérdida Canónica Simpléctica ($\mathcal{L}_{\text{symplectic}}$)

Penaliza el residuo del campo vectorial canónico sobre un lote de tamaño $B$:

$$
\mathcal{L}_{\text{symplectic}}(\theta) = \frac{1}{B} \sum_{k=1}^B \left( \left\| \left. \frac{\partial H_{\theta}}{\partial p} \right|_{(q_k, p_k)} - \dot{q}_k \right\|_2^2 + \left\| \left. \frac{\partial H_{\theta}}{\partial q} \right|_{(q_k, p_k)} + \dot{p}_k \right\|_2^2 \right)
$$

El signo positivo en $\left\| \frac{\partial H_\theta}{\partial q} + \dot{p}_k \right\|_2^2$ proviene de $\dot{p} = -\frac{\partial H}{\partial q} \iff \frac{\partial H}{\partial q} + \dot{p} = \mathbf{0}$.

### B. Pérdida de Conservación de Energía ($\mathcal{L}_{\text{energy}}$)

Fija el término constante arbitrario del potencial respecto a la energía de referencia $H_{0,k}$:

$$
\mathcal{L}_{\text{energy}}(\theta) = \frac{1}{B} \sum_{k=1}^B \left( H_{\theta}(q_k, p_k) - H_{0,k} \right)^2
$$

En la configuración por defecto de `SymplecticPINNLoss`, $\lambda_E = 0.01$ (`energy_weight`).

### C. Extensión Opcional de Momento Angular ($\mathcal{L}_{\text{angular}}$)

Para sistemas con simetría rotacional en coordenadas libres puede incorporarse un término adicional de momento angular $\vec{L}$:

$$
\mathcal{L}_{\text{angular}}(\theta) = \frac{1}{B} \sum_{k=1}^B \left\| \vec{L}(q_k, p_k) - \vec{L}_{0,k} \right\|_2^2, \qquad \vec{L}(q_k, p_k) = \sum_{i=1}^N \vec{r}_{i,k} \times \vec{p}_{i,k}
$$

---

## 4. Preservación del Volumen de Fases: Teorema de Liouville

Una transformación discreta $\mathbf{z}_{t+\Delta t} = \Psi_{\Delta t}(\mathbf{z}_t)$ es simpléctica si su matriz Jacobiana $J = \frac{\partial \mathbf{z}_{t+\Delta t}}{\partial \mathbf{z}_t}$ satisface:

$$
J^T \Omega J = \Omega
$$

Tomando determinantes y considerando $\det(\Omega) = 1$:

$$
\det\left(J^T \Omega J\right) = \det(\Omega) \implies (\det J)^2 = 1
$$

Por continuidad respecto a $J(0) = I$, se verifica $\det(J) = 1$.

### Demostración para el Paso de Euler Simpléctico Separable

Cuando $H_\theta(q, p) = T_\theta(p) + V_\theta(q)$, $\frac{\partial H_\theta}{\partial q} = \nabla_q V_\theta(q)$ y $\frac{\partial H_\theta}{\partial p} = \nabla_p T_\theta(p)$. El esquema semi-implícito implementado en `symplectic_euler_step` propaga el estado mediante:

$$
\begin{aligned}
p_{n+1} &= p_n - \Delta t \, \nabla_q V_\theta(q_n) \\
q_{n+1} &= q_n + \Delta t \, \nabla_p T_\theta(p_{n+1})
\end{aligned}
$$

La transformación $\Phi: (q_n, p_n) \mapsto (q_{n+1}, p_{n+1})$ se factoriza en dos aplicaciones triangulares:

1. **Sub-paso 1 (Actualización de momento):**

   $$
   \Phi_1: \begin{pmatrix} q_n \\ p_n \end{pmatrix} \mapsto \begin{pmatrix} q_n \\ p_n - \Delta t \nabla_q V_\theta(q_n) \end{pmatrix}, \qquad J_1 = \begin{pmatrix} I & 0 \\ -\Delta t \nabla_q^2 V_\theta(q_n) & I \end{pmatrix} \implies \det(J_1) = 1
   $$

2. **Sub-paso 2 (Actualización de posición):**

   $$
   \Phi_2: \begin{pmatrix} q_n \\ p_{n+1} \end{pmatrix} \mapsto \begin{pmatrix} q_n + \Delta t \nabla_p T_\theta(p_{n+1}) \\ p_{n+1} \end{pmatrix}, \qquad J_2 = \begin{pmatrix} I & \Delta t \nabla_p^2 T_\theta(p_{n+1}) \\ 0 & I \end{pmatrix} \implies \det(J_2) = 1
   $$

Por la regla de la cadena, $J = J_2 J_1$:

$$
J = \begin{pmatrix} I - \Delta t^2 (\nabla_p^2 T_\theta)(\nabla_q^2 V_\theta) & \Delta t \nabla_p^2 T_\theta \\ -\Delta t \nabla_q^2 V_\theta & I \end{pmatrix}, \qquad \det(J) = \det(J_2)\det(J_1) = 1
$$

Con $A = \nabla_q^2 V_\theta = A^T$ y $B = \nabla_p^2 T_\theta = B^T$ (matrices hessianas simétricas):

$$
J^T \Omega J = \begin{pmatrix} I - \Delta t^2 AB & -\Delta t A \\ \Delta t B & I \end{pmatrix} \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix} \begin{pmatrix} I - \Delta t^2 BA & \Delta t B \\ -\Delta t A & I \end{pmatrix} = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix} = \Omega
$$

En consecuencia, la 2-forma simpléctica y el volumen del espacio de fases se conservan de forma exacta en la propagación discreta.

---

## 5. Correspondencia en el Repositorio

1. [`pinn_surrogate/model.py`](../../pinn_surrogate/model.py): Define `HamiltonianNN`, `SymplecticPINNLoss`, `symplectic_euler_step` y `rollout_surrogate`.
2. [`pinn_surrogate/tests/test_pinn_conservation.py`](../../pinn_surrogate/tests/test_pinn_conservation.py): Suite `pytest` que verifica autodiferenciación contra diferencias finitas, retropropagación de pérdida y conservación energética acotada en trayectorias de 1,000 pasos.
