#set document(
  title: "Reporte: Implementaciones del Perceptrón",
  author: "Ingeniería de Sistemas Computacionales",
)
#set page(
  paper: "us-letter",
  numbering: "1",
  margin: (x: 2.5cm, y: 2.5cm),
)
#set text(lang: "es", size: 11pt)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => [
  #pagebreak(weak: true)
  #block(above: 1.2em, below: 0.8em)[#text(size: 18pt, weight: "bold")[#it]]
]
#show heading.where(level: 2): it => block(above: 1em, below: 0.5em)[
  #text(size: 13pt, weight: "bold")[#it]
]

// ====== Portada ======
#align(center)[
  #v(3cm)
  #text(size: 22pt, weight: "bold")[
    Reporte comparativo
  ]

  #v(0.4cm)
  #text(size: 18pt, weight: "bold")[
    Implementaciones del Perceptrón
  ]

  #v(0.5cm)
  #text(size: 13pt)[
    Clásico, Descenso por Gradiente (Adaline) y Particle Swarm Optimization
  ]

  #v(2.5cm)
  #text(size: 12pt)[Ingeniería de Sistemas Computacionales]

  #v(0.3cm)
  #text(size: 11pt)[Aprendizaje Automático]

  #v(1.5cm)
  #text(size: 11pt)[2026]
]

#pagebreak()

// ====== Indice ======
#outline(title: [Índice], depth: 2)

// ====== 1. Introduccion ======
= Introducción

El perceptrón es el modelo más sencillo que se puede llamar, sin
exagerar, una "neurona artificial". Lo propuso Frank Rosenblatt en 1958
y casi todas las redes neuronales modernas empiezan reciclando su idea:
sumar entradas con pesos, aplicar una función de activación y comparar
contra una etiqueta. Para una carrera de Sistemas Computacionales sirve
para ver, con poco código, cómo un programa pasa de tener reglas
escritas a mano a derivarlas de los datos.

Este reporte compara tres maneras de entrenar el mismo perceptrón sobre
el mismo problema (clasificación binaria de tumores: maligno vs.
benigno, dataset _Breast Cancer Wisconsin_ de UCI / Kaggle):

+ Perceptrón clásico con la regla delta de Rosenblatt.
+ Perceptrón con descenso por gradiente, en la variante Adaline.
+ Perceptrón con Particle Swarm Optimization (PSO), sin gradientes.

La idea es que quien lo lea entienda primero las matemáticas y después
qué cambia entre cada estrategia: velocidad, estabilidad, exactitud y
costo.

= Marco teórico

== El perceptrón como clasificador lineal

Dado un vector de entrada $bold(x) = (x_1, x_2, dots, x_n)^T in RR^n$, el
perceptrón calcula una salida lineal (también llamada _net input_) como
la suma ponderada de las entradas más un sesgo $b$:

$ z(bold(x)) = sum_(i=1)^(n) w_i x_i + b = bold(w)^T bold(x) + b $

Después aplica una función de activación $phi(dot)$ a $z$. En el
perceptrón clásico esa función es el escalón de Heaviside:

$ phi(z) = cases(
  1 quad &"si " z > 0,
  0 quad &"si " z <= 0,
) $

Geométricamente, $bold(w)^T bold(x) + b = 0$ es un hiperplano en
$RR^n$ que parte el espacio en dos. Lo que cae de un lado es clase 1, lo
del otro es clase 0. Por eso el perceptrón sólo resuelve problemas
_linealmente separables_: necesita que exista al menos un hiperplano
capaz de dejar todos los positivos a un lado y todos los negativos al
otro. Si los datos están entreverados (el clásico XOR, por ejemplo),
ningún ajuste de pesos lo va a salvar.

#figure(
  rect(width: 70%, height: 4cm, stroke: 0.5pt + gray)[
    #align(center + horizon)[
      $bold(w)^T bold(x) + b = 0$ \
      _hiperplano de decisión_
    ]
  ],
  caption: [El perceptrón aprende un hiperplano que separa las dos
    clases en el espacio de características.],
)

== Aprendizaje supervisado y función de pérdida

Sea un conjunto de entrenamiento
$D = {(bold(x)^((k)), y^((k)))}_(k=1)^(N)$ con etiquetas
$y^((k)) in {0, 1}$. Aprender es encontrar el vector
$bold(w)^* in RR^(n+1)$ (incluyendo el bias) que minimiza alguna
función de pérdida $J(bold(w))$. Ahí es donde se separan las tres
implementaciones de este reporte: cada una minimiza algo distinto, con
un algoritmo distinto.

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt,
  align: left,
  table.header(
    [*Implementación*], [*Función de pérdida*], [*Optimizador*],
  ),
  [Clásico],
  [Errores de clasificación (regla delta)],
  [Actualización online],

  [GD / Adaline],
  [Suma de errores al cuadrado (SSE)],
  [Descenso por gradiente batch],

  [PSO],
  [Número de muestras mal clasificadas],
  [Enjambre de partículas],
)

== Regla delta del perceptrón clásico

Para cada muestra $(bold(x), y)$ se calcula la predicción
$hat(y) = phi(z)$ y se actualizan los pesos sólo si hubo error:

$ Delta w_i = eta (y - hat(y)) x_i $
$ w_i <- w_i + Delta w_i $

con $eta in (0, 1]$ la tasa de aprendizaje. La regla es bastante
intuitiva: si la red predijo 0 y debía ser 1, los pesos se mueven en la
dirección de $bold(x)$; si predijo 1 y debía ser 0, se mueven en
sentido opuesto. El bias se actualiza con $Delta b = eta(y - hat(y))$.

Rosenblatt demostró que, si los datos son linealmente separables, esta
regla converge en un número finito de pasos (Teorema de Convergencia del
Perceptrón). Si no lo son, oscila para siempre.

== Descenso por gradiente (Adaline)

La variante Adaline (Widrow & Hoff, 1960) cambia algo aparentemente
pequeño pero con consecuencias grandes: durante el entrenamiento usa la
salida lineal $phi(z) = z$ en lugar del escalón, y define una pérdida
diferenciable, la suma de errores al cuadrado:

$ J(bold(w)) = 1/2 sum_(k=1)^(N) (y^((k)) - phi(z^((k))))^2 $

Como $J$ es diferenciable, se puede calcular su gradiente respecto a
cada peso:

$ (diff J) / (diff w_i) = - sum_(k=1)^(N) (y^((k)) - phi(z^((k)))) x_i^((k)) $

y aplicar descenso por gradiente en sentido opuesto:

$ bold(w) <- bold(w) - eta nabla J(bold(w)) = bold(w) + eta sum_k (y^((k)) - z^((k))) bold(x)^((k)) $

En la implementación se entrena con etiquetas en ${-1, +1}$ y no en
${0, 1}$. La razón: el umbral de decisión es $z > 0$. Con etiquetas
simétricas el gradiente empuja la salida lineal hacia el lado correcto
del cero; con ${0, 1}$ tira hacia $0.5$, que no es la frontera. La suma
también se promedia por $N$ para que $eta$ no dependa del tamaño del
lote.

== Particle Swarm Optimization (PSO)

PSO es un algoritmo metaheurístico inspirado en bandadas de aves. En
vez de calcular un gradiente se mantiene un enjambre de $P$ partículas:
cada partícula $i$ es un candidato a vector de pesos $bold(w)^((i))$
moviéndose por el espacio de búsqueda con velocidad $bold(v)^((i))$.

Cada partícula recuerda dos cosas:
- $bold(p)^((i))$: su mejor posición personal hasta ahora (_pbest_).
- $bold(g)$: la mejor posición que encontró el enjambre completo
  (_gbest_).

En cada iteración, la velocidad se actualiza así:

$ bold(v)^((i)) <- omega bold(v)^((i)) + c_1 r_1 (bold(p)^((i)) - bold(w)^((i))) + c_2 r_2 (bold(g) - bold(w)^((i))) $

y la posición:

$ bold(w)^((i)) <- bold(w)^((i)) + bold(v)^((i)) $

donde $omega$ es la inercia (cuánta velocidad anterior conserva la
partícula), $c_1$ el coeficiente cognitivo (atracción a su mejor
personal), $c_2$ el coeficiente social (atracción al mejor global) y
$r_1, r_2 ~ "Uniforme"(0, 1)$ son ruido estocástico. La función a
minimizar (_fitness_) es directamente el número de muestras mal
clasificadas. No necesita ser diferenciable, lo cual es la gracia.

== Normalización de las entradas

Las tres implementaciones aplican `StandardScaler`, que lleva cada
característica $x_i$ a media cero y desviación estándar uno:

$ tilde(x)_i = (x_i - mu_i) / sigma_i $

Es necesario porque las características viven en escalas muy
diferentes. En el dataset, por ejemplo, el área de la célula está en
cientos y la textura en unidades. Sin normalizar, las features grandes
dominan los pesos sólo por su escala, no porque discriminen mejor.

= Desarrollo: implementaciones y pruebas

== Datos y protocolo experimental

Dataset: _Breast Cancer Wisconsin (Diagnostic)_, descargado vía
`kagglehub` desde `uciml/breast-cancer-wisconsin-data`. Son 569
muestras de núcleos celulares con 30 características numéricas (radio
medio, textura, perímetro, área, suavidad, etc.) y una etiqueta binaria:
`M` (maligno, codificado como 1) o `B` (benigno, codificado como 0). Se
descarta la columna `id` y la columna fantasma `Unnamed: 32` (toda NaN).

Protocolo: las tres implementaciones reciben los mismos $bold(X)$,
$bold(y)$, comparten el `StandardScaler` y la misma semilla aleatoria
(`random_seed = 1`) para que la comparación sea reproducible. La
métrica es el rendimiento por resustitución, es decir, exactitud sobre
el mismo conjunto con el que se entrenó. Mide capacidad de ajuste, no
generalización; eso debe quedar claro porque cualquier modelo se ve
mejor de lo que es cuando se evalúa sobre lo que ya vio.

== Implementación 1: Perceptrón clásico

`Perceptron.py` define la clase base. Los puntos clave:

- `rule(X)` calcula $z = bold(w)^T bold(x) + b$ con
  `np.dot(X, w[1:]) + w[0]`. El bias se guarda en `w[0]`.
- `predecir(X)` aplica el escalón:
  `np.where(rule(X_scaled) > 0, 1, 0)`.
- `entrenar(X, y)` inicializa los pesos con $cal(N)(0, 0.01^2)$,
  recorre `epochs` épocas y dentro de cada época itera muestra por
  muestra aplicando la regla delta sólo cuando hay error.

Pseudocódigo del bucle de entrenamiento:

```python
for _ in range(epochs):
    for xi, target in zip(X_scaled, y):
        predicted = step(rule(xi))
        if target == predicted:
            continue
        update = eta * (target - predicted)
        w[1:] += update * xi
        w[0]  += update
```

Características:
- Actualización online (muestra por muestra).
- Etiquetas en ${0, 1}$.
- Sin función de costo continua, así que no hay curva de convergencia
  suave para diagnosticar.
- Sólo converge si los datos son linealmente separables; si no, oscila.

== Implementación 2: Perceptrón con Descenso por Gradiente

`PerceptronGD.py` hereda de `Perceptron` y sobreescribe `entrenar` con
una variante Adaline:

```python
y_signed = np.where(y == 1, 1, -1)
for _ in range(epochs):
    net_input = self.rule(X_scaled)
    errores   = y_signed - net_input
    w[1:] += eta * X_scaled.T.dot(errores) / n
    w[0]  += eta * errores.sum() / n
    costo = (errores ** 2).sum() / 2.0
    costos.append(costo)
    if abs(costos[-2] - costos[-1]) < tol:
        break
```

Diferencias respecto al clásico:

+ *Error continuo.* Usa $y - z$ (salida lineal), no $y - phi(z)$. El
  gradiente queda bien definido y se sigue mejorando los pesos incluso
  cuando la predicción binaria ya es correcta.
+ *Actualización batch.* El gradiente se promedia sobre todas las
  muestras por época, en lugar de actualizar muestra por muestra.
+ *Etiquetas en ${-1, +1}$.* Simétricas respecto al umbral $z = 0$.
+ *Costo SSE registrado.* Permite graficar convergencia y aplicar
  _early stopping_ cuando $|J_(t-1) - J_t| < "tol"$.
+ *$eta$ más pequeña.* Como el batch acumula muchas muestras,
  `eta = 0.01` es típico (vs. `0.1` del clásico).

== Implementación 3: Perceptrón con PSO

`PerceptronPSO.py` también hereda de `Perceptron`, pero ignora `eta` y
sustituye el entrenamiento por una búsqueda de enjambre:

```python
particulas  = rgen.normal(0, 0.5, (n_particulas, dim))
velocidades = rgen.normal(0, 0.1, (n_particulas, dim))
pbest       = particulas.copy()
pbest_fit   = [fitness(p, X, y) for p in particulas]
gbest       = pbest[argmin(pbest_fit)]

for _ in range(epochs):
    r1, r2 = rand(...), rand(...)
    velocidades = w_inercia*velocidades \
                + c1*r1*(pbest - particulas) \
                + c2*r2*(gbest - particulas)
    particulas += velocidades
    # actualizar pbest/gbest
```

donde `fitness(w, X, y)` cuenta las muestras mal clasificadas.

Diferencias clave:

+ *Sin gradientes.* La función de fitness puede ser discontinua (un
  conteo de errores), cosa que GD no soporta.
+ *Búsqueda poblacional.* 30 partículas exploran el espacio en
  paralelo. Hay menos riesgo de quedarse atrapado en un mínimo local
  estrecho.
+ *Hiperparámetros distintos.* Inercia $omega = 0.7$,
  $c_1 = c_2 = 1.5$, en lugar de tasa de aprendizaje.
+ *Costo computacional.* Cada iteración evalúa la fitness sobre todas
  las partículas, $approx 30 times$ más cómputo por época que GD.
+ *Sin garantías formales de convergencia*, a diferencia del clásico
  para datos separables.

== Resultados

Los tres modelos se entrenaron con los mismos datos durante 100 épocas
(100 iteraciones para PSO, con `n_particulas = 30`). Resultados de
resustitución sobre las 569 muestras:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, center, center, center),
    table.header(
      [*Modelo*], [*Aciertos*], [*Total*], [*Exactitud*],
    ),
    [Perceptrón clásico], [561], [569], [98.59%],
    [Perceptrón GD (Adaline)], [551], [569], [96.84%],
    [Perceptrón PSO], [558], [569], [98.07%],
  ),
  caption: [Rendimiento por resustitución de las tres implementaciones.],
)

Detalle de convergencia:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  align: left,

  [*Clásico*],
  [Actualización online; sin costo continuo registrado.
   Converge rápido por la simplicidad de la regla.],

  [*GD (Adaline)*],
  [Costo SSE inicial: $approx 292.27$;
   Costo SSE final: $approx 77.71$.
   100 épocas usadas (sin disparar early-stopping con `tol=1e-6`).
   Decrece monotónicamente: 292 → 247 → 213 → 188 → 168 → ... → 77.7],

  [*PSO*],
  [Errores iniciales del enjambre (gbest): 39 mal clasificados.
   Tras 100 iteraciones: 11 mal clasificados.
   Trayectoria: 39 → 37 → 29 → 26 → 24 → ... → 11.
   Estabiliza en 11 errores en las últimas iteraciones.],
)

== Análisis

El perceptrón clásico ganó en aciertos (98.59%), pero conviene leerlo
con cuidado: la regla delta sobre datos casi separables produce un
hiperplano de margen pequeño muy pegado al training set. Eso ayuda a la
métrica por resustitución y suele lastimar la generalización.

GD obtuvo menos aciertos ($-1.75$ puntos respecto al clásico) porque
minimiza un proxy distinto: el SSE sobre la salida lineal. Promedia
errores continuos en lugar de empujar los puntos frontera hasta el
último voto. Lo que pierde en exactitud lo gana en estabilidad: la
curva de costo es interpretable, las garantías de convergencia son
formales, y exactamente este algoritmo, con otra activación y muchas
capas, es lo que hace funcionar a una red profunda.

PSO se quedó a medio punto del clásico (98.07%) sin usar derivadas en
ningún momento. En 100 iteraciones bajó de 39 errores a 11, y luego se
estabilizó. El precio es el cómputo: cada iteración requiere evaluar
todas las partículas sobre todo el dataset, y los hiperparámetros
$omega, c_1, c_2$ no se ajustan solos. Pero la idea de minimizar un
conteo de errores discreto, cosa que el gradiente no puede tocar, es
exactamente lo que hace útil a PSO.

Cuándo conviene cada uno:

- *Clásico*: didáctico, o problemas binarios pequeños y casi
  separables.
- *GD*: cuando se quiere diagnóstico de convergencia, pérdidas
  diferenciables, o como puente conceptual hacia redes profundas.
- *PSO*: cuando la pérdida no es diferenciable o no convexa, o cuando
  se quiere explorar globalmente el espacio sin comprometerse a un
  único punto inicial.

= Conclusiones

El perceptrón es el mismo modelo en los tres casos: un hiperplano
$bold(w)^T bold(x) + b = 0$ que parte el espacio en dos clases. Lo que
cambia es _cómo_ se encuentra $bold(w)$, y eso ya cambia bastante.

En este dataset las tres variantes quedan en una franja estrecha,
entre 96.84% y 98.59% de aciertos por resustitución. El clásico encabeza
(561/569), PSO queda muy cerca (558/569) y GD un poco abajo (551/569).
Para clasificación binaria sobre datos casi separables, los tres
funcionan; la diferencia real entre ellos no es la métrica.

GD pierde algo de exactitud, pero a cambio entrega una pérdida
diferenciable, una curva de costo monotónica, y _early stopping_
confiable. Por eso es el ancestro directo del entrenamiento por
backpropagation. PSO demuestra algo distinto: que se puede entrenar el
mismo modelo optimizando una métrica discreta, sin derivadas, a costa
de muchas más evaluaciones por época. El clásico, con su regla delta
mínima, sigue siendo la referencia teórica.

La conclusión práctica para Sistemas Computacionales es que la
arquitectura del modelo y el algoritmo de entrenamiento son
independientes. Mismo perceptrón, tres rutinas distintas, tres
trayectorias distintas, tres conjuntos distintos de garantías. Saber
elegir cuál usar en cada caso es buena parte del trabajo.

#v(1cm)

#align(center)[
  #text(size: 9pt, fill: gray)[
    Reporte generado a partir de las implementaciones en
    `Perceptron.py`, `PerceptronGD.py` y `PerceptronPSO.py`.
  ]
]
