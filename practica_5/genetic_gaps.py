"""
genetic_gaps.py
===============

Algoritmo Genetico (GA) para encontrar la secuencia de gaps que minimiza el
costo de ordenar con Shellsort.

GA en una frase:
  Es una busqueda inspirada en la evolucion biologica. Tenemos una POBLACION de
  soluciones candidatas (cada una = una secuencia de gaps). Las mejores se
  "reproducen" (cruza), sufren pequeñas "mutaciones", y la poblacion mejora
  generacion tras generacion.

Pipeline del GA (lo que hace entrenar()):
  1. Inicializar    -> poblacion aleatoria de cromosomas.
  2. Evaluar        -> medir la aptitud (costo) de cada cromosoma  [_entrenamiento]
  3. Seleccionar    -> elegir padres, favoreciendo a los mejores   [_seleccion_torneo]
  4. Cruzar         -> combinar dos padres para crear hijos        [_cruza_un_punto]
  5. Mutar          -> alterar genes al azar con baja probabilidad  [_mutar]
  6. Elitismo       -> conservar al mejor para no perderlo
  7. Repetir 2-6 durante n_generaciones.

Elegimos este GA "generacional estandar" por ser el mas facil de implementar y
entender: representacion de longitud fija, seleccion por torneo, cruza de un
punto y mutacion por reemplazo de gen.
"""

import numpy as np

from shell_sort import shell_sort
import datos


class AlgoritmoGeneticoGaps:
    """
    Busca, mediante un algoritmo genetico, la secuencia de gaps que ordena un
    arreglo desordenado de tamaño N con el menor costo de Shellsort.

    Codificacion del cromosoma:
      Un cromosoma es un vector de 'k_gaps' enteros en el rango [1, N//2].
      Esos enteros, una vez decodificados (unicos, ordenados de mayor a menor y
      forzando que terminen en 1), forman la secuencia de gaps de Shellsort.
    """

    def __init__(self,
                 n: int,
                 k_gaps: int = 5,
                 n_poblacion: int = 40,
                 n_generaciones: int = 60,
                 prob_cruza: float = 0.8,
                 prob_mutacion: float = 0.1,
                 n_muestras: int = 10,
                 random_seed: int = 1):
        """
        :param n: tamaño del arreglo a ordenar.
        :param k_gaps: cuantos genes (gaps) tiene cada cromosoma.
        :param n_poblacion: cuantos cromosomas hay por generacion.
        :param n_generaciones: cuantas iteraciones evoluciona la poblacion.
        :param prob_cruza: probabilidad de cruzar un par de padres.
        :param prob_mutacion: probabilidad de mutar cada gen.
        :param n_muestras: cuantos reordenamientos del arreglo base se usan
                           para promediar el costo (mide gaps "buenos en general").
        :param random_seed: semilla para reproducibilidad.
        """
        self._n = n
        self._k_gaps = k_gaps
        self._n_poblacion = n_poblacion
        self._n_generaciones = n_generaciones
        self._prob_cruza = prob_cruza
        self._prob_mutacion = prob_mutacion
        self._n_muestras = n_muestras

        # Gap maximo razonable: la mitad del arreglo (mas grande no tiene sentido).
        self._gap_max = max(2, n // 2)

        # Generador aleatorio propio para que TODO el GA sea reproducible.
        self._rgen = np.random.RandomState(random_seed)

        # Arreglo base desordenado (cargado/creado en disco). Sobre el medimos costo.
        self._arreglo_base = datos.cargar_arreglo(n, random_seed=random_seed)

        # Resultados que se llenan al entrenar.
        self.mejores_gaps = None        # mejor secuencia de gaps encontrada
        self.mejor_costo = None         # su costo promedio
        self.historial_fitness = []     # mejor costo por generacion (curva de aprendizaje)

    # ------------------------------------------------------------------
    # Decodificacion: cromosoma (enteros) -> secuencia de gaps valida
    # ------------------------------------------------------------------
    def _decodificar(self, cromosoma):
        """
        Convierte un vector de enteros en una secuencia de gaps VALIDA:
          - Quitamos duplicados (gaps repetidos no aportan pasadas utiles).
          - Ordenamos de mayor a menor (Shellsort va de gaps grandes a chicos).
          - Forzamos que termine en 1: sin el gap=1 el arreglo no queda
            garantizadamente ordenado (la ultima pasada debe ser insercion clasica).
        """
        gaps = sorted(set(int(g) for g in cromosoma), reverse=True)
        if gaps and gaps[-1] != 1:
            gaps.append(1)
        if not gaps:
            gaps = [1]
        return gaps

    # ------------------------------------------------------------------
    # Entrenamiento = funcion de aptitud ("fitness"). MENOR costo = mejor.
    # ------------------------------------------------------------------
    def _entrenamiento(self, cromosoma, arreglos_muestra):
        """
        Mide la aptitud de un cromosoma: el costo PROMEDIO de ordenar varios
        reordenamientos del arreglo base con sus gaps.

        Promediar sobre varias muestras evita que el GA encuentre gaps que solo
        funcionan para una disposicion concreta; buscamos gaps buenos "en general"
        para arreglos de tamaño N.

        :param cromosoma: vector de enteros (genes).
        :param arreglos_muestra: lista de arreglos de prueba (los mismos para
               todos los cromosomas de esta generacion -> comparacion justa).
        :return: costo promedio (float). Cuanto menor, mejor.
        """
        gaps = self._decodificar(cromosoma)
        costo_total = 0
        for arr in arreglos_muestra:
            _, costo = shell_sort(arr, gaps)
            costo_total += costo
        return costo_total / len(arreglos_muestra)

    def _generar_muestras(self):
        """
        Para el caso peor (arreglo en orden inverso), evaluamos siempre
        sobre el arreglo base directamente en lugar de permutaciones.
        """
        return [self._arreglo_base]

    # ------------------------------------------------------------------
    # Operadores geneticos
    # ------------------------------------------------------------------
    def _seleccion_torneo(self, poblacion, costos, tam_torneo=3):
        """
        Seleccion por torneo: se eligen 'tam_torneo' cromosomas al azar y gana
        (se selecciona) el de MENOR costo. Es simple y no necesita escalar el
        fitness; ademas mantiene algo de diversidad (no siempre gana el mejor global).
        """
        indices = self._rgen.randint(0, len(poblacion), size=tam_torneo)
        mejor_idx = min(indices, key=lambda i: costos[i])
        return poblacion[mejor_idx].copy()

    def _cruza_un_punto(self, padre1, padre2):
        """
        Cruza de un punto: se elige una posicion de corte y los hijos heredan la
        primera parte de un padre y la segunda parte del otro. Es la cruza mas
        intuitiva y suficiente para vectores de longitud fija.
        """
        # Si la suerte no favorece la cruza, los hijos son copias de los padres.
        if self._rgen.rand() > self._prob_cruza:
            return padre1.copy(), padre2.copy()

        punto = self._rgen.randint(1, self._k_gaps)  # corte interno (no en los extremos)
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2

    def _mutar(self, cromosoma):
        """
        Mutacion por reemplazo: cada gen, con probabilidad 'prob_mutacion', se
        sustituye por un entero aleatorio nuevo en [1, gap_max]. Esto inyecta
        diversidad y permite explorar gaps que la cruza sola no produciria.
        """
        for i in range(len(cromosoma)):
            if self._rgen.rand() < self._prob_mutacion:
                cromosoma[i] = self._rgen.randint(1, self._gap_max + 1)
        return cromosoma

    # ------------------------------------------------------------------
    # Bucle principal del GA
    # ------------------------------------------------------------------
    def entrenar(self):
        """
        Ejecuta el algoritmo genetico completo y guarda la mejor secuencia de
        gaps encontrada en self.mejores_gaps.
        """
        # 1. Poblacion inicial: n_poblacion cromosomas de k_gaps enteros aleatorios.
        poblacion = [self._rgen.randint(1, self._gap_max + 1, size=self._k_gaps)
                     for _ in range(self._n_poblacion)]

        for generacion in range(self._n_generaciones):
            # Mismas muestras para todos los cromosomas de esta generacion.
            muestras = self._generar_muestras()

            # 2. Evaluar: costo de cada cromosoma.
            costos = [self._entrenamiento(c, muestras) for c in poblacion]

            # Elitismo: localizamos al mejor de la generacion para conservarlo.
            idx_mejor = int(np.argmin(costos))
            elite = poblacion[idx_mejor].copy()
            costo_elite = costos[idx_mejor]

            # Guardamos la mejor solucion historica.
            if self.mejor_costo is None or costo_elite < self.mejor_costo:
                self.mejor_costo = costo_elite
                self.mejores_gaps = self._decodificar(elite)

            self.historial_fitness.append(costo_elite)

            # 3-5. Construimos la nueva poblacion.
            nueva_poblacion = [elite]  # el elite pasa directo (elitismo)
            while len(nueva_poblacion) < self._n_poblacion:
                padre1 = self._seleccion_torneo(poblacion, costos)
                padre2 = self._seleccion_torneo(poblacion, costos)
                hijo1, hijo2 = self._cruza_un_punto(padre1, padre2)
                nueva_poblacion.append(self._mutar(hijo1))
                if len(nueva_poblacion) < self._n_poblacion:
                    nueva_poblacion.append(self._mutar(hijo2))

            # 6. La nueva generacion reemplaza a la anterior.
            poblacion = nueva_poblacion

        return self.mejores_gaps
