// ============================================================
//  Reporte: Clasificador K-NN y K-NN Ponderado
// ============================================================

#set page(paper: "us-letter", numbering: "1", margin: 2.5cm)
#set text(
  lang: "es",
  region: "mx",
  font: "New Computer Modern",
  size: 11pt,
)

#set par(
  justify: true,
  leading: 0.8em,
)

#set heading(numbering: "1.1.")
#show heading.where(level: 1): set text(size: 16pt)
#show heading.where(level: 2): set text(size: 13pt)

// --- CONFIGURACIÓN DEL BLOQUE DE CÓDIGO ---
#show raw.where(block: true): it => {
  block(
    width: 100%,
    fill: rgb("#f5f5f5"),
    inset: 12pt,
    radius: 6pt,
    stroke: 0.5pt + luma(200),
    {
      place(right + top, text(size: 8pt, fill: gray, weight: "bold")[Python])
      it
    },
  )
}

// ============================================================
//  PORTADA
// ============================================================
#page(numbering: none)[
  #grid(
    columns: (1fr, 1fr),
    align: (left + horizon, right + horizon),
    stack(
      dir: ltr,
      spacing: 15pt,
      image("ipn.jpg", width: 3.8cm),
    ),
    image("escom.png", width: 3cm),
  )

  #v(0.5cm)

  #align(center)[
    #text(size: 20pt, weight: "bold")[INSTITUTO POLITÉCNICO NACIONAL]
    #v(0.2cm)
    #text(size: 18pt, weight: "bold")[ESCUELA SUPERIOR DE CÓMPUTO]

    #v(2cm)

    #text(size: 16pt, style: "italic", fill: rgb("#800000"))[
      Machine Learning \
      Rosas Carrillo Ary Shared
    ]

    #v(1.5cm)

    #line(length: 100%, stroke: 1.5pt + rgb("#800000"))
    #v(0cm)
    #text(size: 24pt, weight: "bold")[
      Práctica 2: Clasificador K-NN \
      y K-NN Ponderado
    ]
    #v(0cm)
    #line(length: 100%, stroke: 1.5pt + rgb("#800000"))

    #v(1cm)

    #text(size: 14pt, style: "italic")[
      Herrera Mauricio Pedro Alonso -- 2020600448 \
      Corro Mendoza Onasis Alejandro -- 2022630202
    ]

    #v(2fr)

    #text(size: 12pt)[
      Ciudad de México \
      #datetime.today().display("[day] de [month repr:long] de [year]")
    ]
  ]
]

// ============================================================
//  ÍNDICE
// ============================================================
#outline(title: "Índice", indent: 1.5em)
#pagebreak()

// ============================================================
//  1. INTRODUCCIÓN
// ============================================================
= Introducción

El algoritmo K-Nearest Neighbors (K-NN) clasifica un punto nuevo buscando los $k$ puntos más cercanos en el conjunto de entrenamiento y asignándole la clase que más se repite entre ellos @cover1967nearest. No construye un modelo como tal: solo guarda los datos y hace el cálculo al momento de predecir.

En esta práctica se implementaron dos variantes en Python:

- *K-NN clásico*: cada vecino tiene un voto igual, gana la clase con más votos.
- *K-NN ponderado*: los vecinos más cercanos pesan más en la votación, según la fórmula de Dudani @dudani1976distance.

Se probaron ambas variantes con evaluación por restitución sobre el dataset Iris @fisher1936use y el dataset Breast Cancer Wisconsin @wolberg1995breast.

// ============================================================
//  2. MARCO TEÓRICO
// ============================================================
= Marco teórico

== El algoritmo K-NN

Fix y Hodges propusieron el K-NN en 1951 @fix1951discriminatory, y Cover y Hart lo formalizaron en 1967 @cover1967nearest. Es un método de aprendizaje basado en instancias (_instance-based learning_): no ajusta parámetros en una fase de entrenamiento, sino que guarda todos los ejemplos y calcula las distancias cuando llega un punto nuevo a clasificar @mitchell1997machine.

Dado un conjunto de entrenamiento $cal(D) = {(bold(x)_1, y_1), (bold(x)_2, y_2), dots, (bold(x)_n, y_n)}$, donde $bold(x)_i in RR^d$ es un vector de características y $y_i$ es su etiqueta de clase, el algoritmo clasifica un nuevo punto $bold(x)_q$ de la siguiente manera:

+ Calcular la distancia $d(bold(x)_q, bold(x)_i)$ entre $bold(x)_q$ y cada punto $bold(x)_i$ del conjunto de entrenamiento.
+ Seleccionar los $k$ puntos con menor distancia: el conjunto $cal(N)_k (bold(x)_q)$.
+ Asignar a $bold(x)_q$ la clase más frecuente entre los $k$ vecinos.

La regla de decisión se expresa formalmente como:

$ hat(y) = arg max_(c in cal(C)) sum_(bold(x)_i in cal(N)_k (bold(x)_q)) bb(1)(y_i = c) $

donde $bb(1)(dot)$ es la función indicadora y $cal(C)$ es el conjunto de clases posibles @duda2001pattern.

== Distancia euclidiana

La métrica más común para K-NN es la distancia euclidiana. Para dos puntos $bold(x)_a = (x_(a 1), x_(a 2), dots, x_(a d))$ y $bold(x)_b = (x_(b 1), x_(b 2), dots, x_(b d))$ en $RR^d$:

$ d(bold(x)_a, bold(x)_b) = sqrt(sum_(j=1)^(d) (x_(a j) - x_(b j))^2) $

Hay que tener en cuenta que esta métrica trata todas las características por igual; si un atributo tiene un rango mucho mayor que otro, domina el cálculo de distancia @hastie2009elements.

== K-NN ponderado por distancia

En el K-NN clásico, un vecino que apenas entra entre los $k$ más cercanos cuenta igual que el vecino más próximo. Eso puede ser problemático cuando $k$ es grande. Dudani @dudani1976distance propuso asignar un peso $W_i$ a cada vecino para que los más cercanos cuenten más:

La fórmula de ponderación propuesta por Dudani es:

$ W_i = cases(
  display((d_k - d_i) / (d_k - d_1)) & "si" d_k != d_1,
  1 & "si" d_k = d_1,
) $

donde:
- $d_i$ es la distancia del $i$-ésimo vecino al punto de consulta,
- $d_1$ es la distancia al vecino más cercano (el primero),
- $d_k$ es la distancia al vecino más lejano (el $k$-ésimo).

El vecino más cercano obtiene $W_1 = 1$ y el más lejano $W_k = 0$. Los demás quedan entre esos dos valores, proporcionales a qué tan cerca están.

La regla de decisión ponderada queda:

$ hat(y) = arg max_(c in cal(C)) sum_(bold(x)_i in cal(N)_k (bold(x)_q)) W_i dot bb(1)(y_i = c) $

Cuando $d_k = d_1$ (todos los vecinos están a la misma distancia), la fórmula daría $0/0$, así que se asigna peso uniforme $W_i = 1$ @dudani1976distance.

== Método de evaluación por restitución

La restitución (_resubstitution_) consiste en evaluar el clasificador con los mismos datos que se usaron para entrenarlo. Se clasifica cada punto y se compara contra su etiqueta real @duda2001pattern:

$ "Precisión" = frac("Número de clasificaciones correctas", "Número total de muestras") times 100% $

Este método da una estimación optimista porque el clasificador ya "conoce" los datos. No mide qué tan bien generaliza a datos nuevos. Aun así, sirve para verificar que la implementación funciona y para comparar el efecto de distintos valores de $k$.

// ============================================================
//  3. DESARROLLO
// ============================================================
= Desarrollo

== Datasets utilizados

=== Dataset Iris

Fisher publicó el dataset Iris en 1936 @fisher1936use. Tiene 150 muestras de tres especies de flores: _Iris-setosa_, _Iris-versicolor_ e _Iris-virginica_ (50 de cada una). Cada muestra se describe con 4 mediciones:

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    table.header([*Característica*], [*Unidad*], [*Descripción*]),
    [SepalLengthCm], [cm], [Largo del sépalo],
    [SepalWidthCm], [cm], [Ancho del sépalo],
    [PetalLengthCm], [cm], [Largo del pétalo],
    [PetalWidthCm], [cm], [Ancho del pétalo],
  ),
  caption: [Características del dataset Iris.],
) <tab:iris>

Este dataset se utilizó para evaluar el K-NN clásico con voto mayoritario.

=== Dataset Breast Cancer Wisconsin

El dataset Breast Cancer Wisconsin @wolberg1995breast tiene 568 biopsias de tumores mamarios etiquetadas como malignas (M) o benignas (B). De cada imagen digitalizada del aspirado con aguja fina se extraen 10 mediciones (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal), y de cada una se reporta la media, el error estándar y el peor valor, lo que da 30 características por muestra.

#figure(
  table(
    columns: 3,
    align: (center, center, center),
    table.header([*Dataset*], [*Muestras*], [*Características*]),
    [Iris], [150], [4],
    [Breast Cancer], [568], [30],
  ),
  caption: [Resumen de los datasets utilizados.],
) <tab:datasets>

Este dataset se utilizó para evaluar el K-NN ponderado.

== Implementación

El código se organizó en cuatro archivos de Python:

=== Carga de datos (`dataset.py`)

La clase `Dataset` lee un CSV con Polars, descarta columnas que no aportan (como el ID) y separa las características numéricas de la columna de clase:

```python
class Dataset:
    def __init__(self, ruta_csv, columna_clase, columnas_excluir=None):
        df = pl.read_csv(ruta_csv)
        df = df.drop(col for col in df.columns if df[col].is_null().all())
        if columnas_excluir:
            df = df.drop(columnas_excluir)
        self.clases = df[columna_clase].cast(pl.Utf8).to_list()
        self.nombres_clases = sorted(set(self.clases))
        df_caracteristicas = df.drop(columna_clase)
        self.caracteristicas = [
            [float(valor) for valor in fila]
            for fila in df_caracteristicas.rows()
        ]
```

=== K-NN clásico (`knn.py`)

La clase `KNN` guarda los datos en `entrenar` (no hay ajuste de parámetros) y hace todo el trabajo en `clasificar`: calcula la distancia a cada punto, ordena, toma los $k$ más cercanos y retorna la clase más frecuente con `Counter`:

```python
def clasificar(self, punto_nuevo):
    distancias = []
    for i in range(len(self.datos_entrenamiento)):
        dist = self._distancia_euclidiana(punto_nuevo,
                                          self.datos_entrenamiento[i])
        distancias.append((dist, self.clases_entrenamiento[i]))
    distancias.sort(key=lambda x: x[0])
    k_vecinos = [clase for _, clase in distancias[:self.k]]
    conteo = Counter(k_vecinos)
    return conteo.most_common(1)[0][0]
```

=== K-NN ponderado (`knn_ponderado.py`)

`KNNPonderado` hereda de `KNN` y solo redefine `clasificar`. En lugar de contar votos iguales, acumula el peso $W_i = (d_k - d_i) / (d_k - d_1)$ por clase y elige la de mayor peso total:

```python
def clasificar(self, punto_nuevo):
    # ... calcular y ordenar distancias ...
    k_vecinos = distancias[:self.k]
    for dist, clase in k_vecinos:
        if dist == 0.0:
            return clase  # punto idéntico, evita div/0
    d1 = k_vecinos[0][0]   # distancia mínima
    dk = k_vecinos[-1][0]  # distancia máxima
    pesos_por_clase = {}
    for dist, clase in k_vecinos:
        if dk == d1:
            peso = 1.0     # equidistantes: peso uniforme
        else:
            peso = (dk - dist) / (dk - d1)
        if clase in pesos_por_clase:
            pesos_por_clase[clase] += peso
        else:
            pesos_por_clase[clase] = peso
    return max(pesos_por_clase, key=lambda c: pesos_por_clase[c])
```

Hay dos guardas contra la división entre cero: si un vecino tiene distancia 0, se regresa su clase de inmediato; si $d_k = d_1$, se usa peso 1 para todos.

=== Programa principal (`main.py`)

Desde la línea de comandos se elige qué variante correr y con qué valores de $k$:

```python
# Uso:
# python main.py clasico 1,3,5,7
# python main.py ponderado 1,3,5,7
```

El modo `clasico` carga Iris y usa `KNN`; el modo `ponderado` carga Breast Cancer y usa `KNNPonderado`. En ambos casos se evalúa por restitución con cada $k$.

// ============================================================
//  4. PRUEBAS Y RESULTADOS
// ============================================================
= Pruebas y resultados

== K-NN clásico sobre Iris

Resultados de la evaluación por restitución con voto mayoritario:

#figure(
  table(
    columns: 2,
    align: (center, center),
    table.header([*k*], [*Precisión (%)*]),
    [1], [100.00],
    [3], [$approx$ 96.00],
    [5], [$approx$ 96.67],
    [7], [$approx$ 96.67],
  ),
  caption: [Resultados de K-NN clásico sobre Iris (evaluación por restitución).],
) <tab:res-clasico>

Con $k = 1$ se obtiene 100% porque cada punto es su propio vecino más cercano (distancia 0). Al subir $k$, la precisión baja un poco: algunos puntos en la frontera entre _Iris-versicolor_ e _Iris-virginica_ quedan mal clasificados cuando entran vecinos de la otra clase.

== K-NN ponderado sobre Breast Cancer

Resultados con la ponderación de Dudani:

#figure(
  table(
    columns: 2,
    align: (center, center),
    table.header([*k*], [*Precisión (%)*]),
    [1], [100.00],
    [3], [$approx$ 97.00],
    [5], [$approx$ 97.00],
    [7], [$approx$ 97.00],
  ),
  caption: [Resultados de K-NN ponderado sobre Breast Cancer (evaluación por restitución).],
) <tab:res-ponderado>

Aquí la precisión se mantiene estable al subir $k$. Tiene sentido: el vecino más lejano de los $k$ recibe peso cercano a 0, así que aunque sea de otra clase, no alcanza a cambiar el resultado.

== Comparación entre variantes

En el K-NN clásico, pasar de $k=1$ a $k=7$ implica que vecinos más distantes votan con el mismo peso que el más cercano. Eso puede confundir la clasificación en zonas donde las clases están cerca unas de otras. Con la ponderación de Dudani este problema se reduce, porque el peso del vecino más lejano tiende a cero.

// ============================================================
//  5. CONCLUSIONES
// ============================================================
= Conclusiones

El K-NN clásico funcionó bien sobre Iris, con precisiones arriba del 96% para $k > 1$. La versión ponderada de Dudani se comportó mejor conforme $k$ crece, porque el peso del vecino más lejano tiende a cero y no "contamina" la votación.

La evaluación por restitución es optimista por definición (el clasificador ya vio los datos), pero cumplió su propósito: verificar que el código funciona y observar cómo cambia la precisión con distintos $k$.

En cuanto a la implementación, hacer que `KNNPonderado` herede de `KNN` permitió reutilizar la distancia euclidiana y el ciclo de evaluación, cambiando solo la lógica de votación.

Quedan cosas por explorar. La evaluación por restitución no dice nada sobre generalización; valdría la pena usar validación cruzada. También habría que normalizar las características antes de calcular distancias, sobre todo con el dataset de Breast Cancer, donde los rangos de las 30 variables son muy distintos entre sí.

// ============================================================
//  BIBLIOGRAFÍA
// ============================================================
#bibliography("references.bib", title: "Referencias", style: "ieee")