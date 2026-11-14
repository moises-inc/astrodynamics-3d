# Nota Técnica: Shaders 3D WebGL, Atmósferas y Post-Procesamiento HDR

Este documento describe las formulaciones analíticas y físicas de los sombreadores GLSL y el pipeline de post-procesamiento implementados con React 19, Three.js y WebGL en [`web_ui/src/visuals/`](../../web_ui/src/visuals).

---

## 1. Shaders Solares y Atmósferas Planetarias

### A. Dispersión Atmosférica de Rayleigh (Fresnel Invertido)

El material atmosférico ([`AtmosphereShader.ts`](../../web_ui/src/visuals/AtmosphereShader.ts)) modela el halo en el limbo planetario sobre una esfera escalada ($R_{\text{atm}} = 1.18 \, R_{\text{planeta}}$) renderizando caras posteriores (`side: THREE.BackSide`) con mezcla aditiva (`THREE.AdditiveBlending`).

La intensidad marginal depende del producto escalar entre la normal superficial $\vec{n}$ y la dirección de vista $\vec{v}$ ($\|\vec{n}\| = \|\vec{v}\| = 1$):

$$
F_{\text{Fresnel}} = 1.0 - \max(\vec{n} \cdot \vec{v}, 0.0)
$$

$$
\alpha = I_0 \left( F_{\text{Fresnel}} \right)^p, \qquad \mathbf{C}_{\text{out}} = 1.5 \, \mathbf{C}_{\text{atmósfera}}
$$

Donde:
- $\vec{n} \cdot \vec{v} = \cos\theta$: En el centro del disco ($\theta = 0$), $\alpha = 0$; en el limbo rasante ($\theta \to \pi/2$), $\alpha \to I_0$.
- $p = 3.5$ es el exponente de decaimiento (`uPower`).
- $I_0 = 1.25$ es el factor de intensidad (`uIntensity`).

### B. Granulación Fotosférica Solar con Ruido Simplex 3D

El shader solar ([`SunShader.ts`](../../web_ui/src/visuals/SunShader.ts)) sintetiza turbulencia convectiva de plasma y realce coronal en el limbo.

#### 1. Campo de Ruido Multifrecuencia

A partir de la coordenada local del vértice $\vec{r} \in \mathbb{R}^3$ (`vPosition`) y el tiempo escalado $t' = 0.45 t$ (`uTime * 0.45`), se evalúan dos octavas de ruido Simplex 3D $\text{snoise}: \mathbb{R}^3 \to [-1, 1]$:

$$
N_1(\vec{r}, t) = \text{snoise}\left( 3.5 \vec{r} + \begin{pmatrix} 0 \\ 0.8 t' \\ 0.4 t' \end{pmatrix} \right)
$$

$$
N_2(\vec{r}, t) = \text{snoise}\left( 8.0 \vec{r} - \begin{pmatrix} 0.5 t' \\ 0 \\ 0.7 t' \end{pmatrix} \right)
$$

El escalar normalizado en $[0, 1]$ combina ambas octavas:

$$
s(\vec{r}, t) = \frac{1}{2} \left[ N_1(\vec{r}, t) + \frac{1}{2} N_2(\vec{r}, t) \right] + \frac{1}{2} = \frac{1}{2} N_1(\vec{r}, t) + \frac{1}{4} N_2(\vec{r}, t) + \frac{1}{2}
$$

#### 2. Interpolación de Plasma

El color fotosférico interpola entre el tono base $\mathbf{C}_{\text{base}}$ (`#fbbf24`), el núcleo $\mathbf{C}_{\text{core}}$ (`#ffffff`) y la componente cromosférica $\mathbf{C}_{\text{flare}}$ (`#ef4444`):

$$
\mathbf{C}_1(s) = \left(1 - s^{1.8}\right) \mathbf{C}_{\text{base}} + s^{1.8} \mathbf{C}_{\text{core}}
$$

$$
w_{\text{flare}}(s) = 0.65 (1.0 - s)
$$

$$
\mathbf{C}_{\text{plasma}}(s) = \left(1 - w_{\text{flare}}(s)\right) \mathbf{C}_1(s) + w_{\text{flare}}(s) \mathbf{C}_{\text{flare}}
$$

#### 3. Realce de Limbo Solar y Emisión Final

$$
F_{\text{rim}} = 1.0 - \max(\vec{n} \cdot \vec{v}, 0.0)
$$

$$
\mathbf{C}_{\text{rim}} = 1.8 \left(F_{\text{rim}}\right)^{2.2} \mathbf{C}_{\text{base}} + \left(F_{\text{rim}}\right)^{4.0} \mathbf{C}_{\text{core}}
$$

$$
\mathbf{C}_{\text{final}} = 1.4 \left( \mathbf{C}_{\text{plasma}}(s) + \mathbf{C}_{\text{rim}} \right)
$$

Adicionalmente, `createSunCoronaMesh` añade una capa envolvente de radio $1.45 \, R$ con modulación armónica $\text{pulse}(t) = 0.85 + 0.15 \sin(2.5 t)$.

---

## 2. Campo Estelar Volumétrico y Clasificación Espectral de Harvard

El módulo [`CosmicStarfield.ts`](../../web_ui/src/visuals/CosmicStarfield.ts) distribuye 5,500 partículas estelares en una cáscara esférica mediante muestreo uniforme sobre la esfera unitaria:

$$
r = 180 + 320 u_r, \qquad u_r \sim \mathcal{U}(0, 1) \implies r \in [180, 500]
$$

$$
\theta = 2\pi u_\theta, \qquad u_\theta \sim \mathcal{U}(0, 1)
$$

$$
\phi = \arccos(2u_\phi - 1), \qquad u_\phi \sim \mathcal{U}(0, 1) \implies \cos\phi \sim \mathcal{U}(-1, 1)
$$

$$
\vec{r}_{\text{star}} = \begin{pmatrix} r \sin\phi \cos\theta \\ r \sin\phi \sin\theta \\ r \cos\phi \end{pmatrix}
$$

### A. Sombreadores de Partícula Estelar

En el *Vertex Shader*, el tamaño proyectado incorpora atenuación por perspectiva y centelleo armónico individual:

$$
S_{\text{screen}} = S_0 \cdot \alpha_{\text{twinkle}}(t, \psi) \left( \frac{300}{-z_{\text{view}}} \right)
$$

$$
\alpha_{\text{twinkle}}(t, \psi) = 0.75 + 0.25 \sin(2.8 t + 4.0 \psi), \qquad \psi \sim \mathcal{U}(0, 2\pi)
$$

En el *Fragment Shader*, cada punto se atenúa radialmente con perfil gaussiano continuo dentro del disco unitario:

$$
d = \left\| \vec{u}_{\text{coord}} - \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix} \right\|_2
$$

$$
I(d) = \begin{cases} \exp(-10 d^2), & \text{si } d \le 0.5 \\ 0, & \text{si } d > 0.5 \text{ (discard)} \end{cases}
$$

$$
\mathbf{C}_{\text{frag}} = 1.3 \, \mathbf{C}_{\text{espectral}} \cdot I(d) \cdot \alpha_{\text{twinkle}}
$$

### B. Secuencia de Harvard y Calibración Cromática

| Clase | Temperatura ($T_{\text{eff}}$) | Color Hex | Tipo Espectral | Probabilidad en Simulación | Abundancia Astrofísica Aprox. |
| :---: | :---: | :---: | :--- | :---: | :---: |
| **O** | $35{,}000\text{ K}$ | `#9bb0ff` | Azul Profundo | $5.0\%$ ($p < 0.05$) | $\sim 0.00003\%$ |
| **B** | $20{,}000\text{ K}$ | `#bbccff` | Blanco Azulado | $10.0\%$ ($0.05 \le p < 0.15$) | $\sim 0.13\%$ |
| **A** | $9{,}000\text{ K}$ | `#f8f9ff` | Blanco Puro | $15.0\%$ ($0.15 \le p < 0.30$) | $\sim 0.60\%$ |
| **F** | $7{,}000\text{ K}$ | `#fff4ea` | Blanco Amarillento | $20.0\%$ ($0.30 \le p < 0.50$) | $\sim 3.00\%$ |
| **G** | $5{,}500\text{ K}$ | `#fff2a3` | Amarillo Solar | $20.0\%$ ($0.50 \le p < 0.70$) | $\sim 7.60\%$ |
| **K** | $4{,}000\text{ K}$ | `#ffd2a1` | Naranja Cálido | $18.0\%$ ($0.70 \le p < 0.88$) | $\sim 12.10\%$ |
| **M** | $3{,}000\text{ K}$ | `#ff8f70` | Enana Roja / Gigante | $12.0\%$ ($p \ge 0.88$) | $\sim 76.40\%$ |

Nota: La simulación incrementa la proporción de estrellas luminosas O, B y A ($30\%$ combinado) para preservar variedad cromática en escena frente a la distribución estelar real.

---

## 3. Pipeline HDR: Bloom Volumétrico y Mapeo de Tonos ACES

El pipeline de post-procesamiento en [`OrbitCanvas3D.tsx`](../../web_ui/src/components/OrbitCanvas3D.tsx) encadena `EffectComposer` con `UnrealBloomPass` y mapeo de tonos `ACESFilmicToneMapping`.

### 1. Extracción de Luminancia por Umbral

Dada la radiancia lineal $\mathbf{C} = (R, G, B)$ en espacio HDR, la luminancia ITU-R BT.709 es:

$$
Y = 0.2126 R + 0.7152 G + 0.0722 B
$$

Con umbral de bloom $Y_{\text{th}} = 0.18$:

$$
k(Y) = \frac{\max(Y - Y_{\text{th}}, 0)}{Y + 10^{-4}}, \qquad \mathbf{C}_{\text{bright}} = k(Y) \mathbf{C}
$$

### 2. Convolución Gaussiana 2D Separable

El desenfoque multipaso utiliza la factorización separable del núcleo gaussiano:

$$
G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right) = \left( \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}} \right) \left( \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{y^2}{2\sigma^2}} \right)
$$

reduciendo el costo por píxel de $\mathcal{O}(K^2)$ a $\mathcal{O}(2K)$ accesos a textura.

### 3. Composición HDR y Curva ACES Filmic

$$
\mathbf{C}_{\text{HDR}} = \mathbf{C}_{\text{escena}} + \beta \cdot \mathbf{C}_{\text{bloom}}
$$

donde $\beta \in [0.2, 2.5]$ (`bloomIntensity`, por defecto `1.2`). Finalmente, `ACESFilmicToneMapping` con exposición $E = 1.1$ comprime el rango dinámico mediante la aproximación racional de Narkowicz:

$$
\mathbf{x} = 1.1 \, \mathbf{C}_{\text{HDR}}, \qquad \mathbf{C}_{\text{SDR}} = \frac{\mathbf{x}(2.51 \mathbf{x} + 0.03)}{\mathbf{x}(2.43 \mathbf{x} + 0.59) + 0.14}
$$
