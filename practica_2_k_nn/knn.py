import math  # módulo para funciones matemáticas como raíz cuadrada
from collections import Counter  # clase para contar elementos y obtener los más frecuentes


class KNN:
    """Implementación del algoritmo K-Nearest Neighbors (K vecinos más cercanos)."""

    def __init__(self, k: int = 5):
        # guardar el número de vecinos a considerar para la clasificación
        self.k: int = k
        # inicializar los datos de entrenamiento como listas vacías
        self.datos_entrenamiento: list[list[float]] = []
        # inicializar las clases de entrenamiento como lista vacía
        self.clases_entrenamiento: list[str] = []

    def entrenar(self, caracteristicas: list[list[float]], clases: list[str]) -> None:
        """Almacena los datos de entrenamiento (KNN no entrena un modelo, solo guarda los datos)."""
        # guardar las características de entrenamiento (cada elemento es un punto n-dimensional)
        self.datos_entrenamiento = caracteristicas
        # guardar la clase correspondiente a cada punto de entrenamiento
        self.clases_entrenamiento = clases

    def _distancia_euclidiana(self, punto_a: list[float], punto_b: list[float]) -> float:
        """Calcula la distancia euclidiana entre dos puntos n-dimensionales."""
        # inicializar el acumulador de la suma de cuadrados
        suma_cuadrados = 0.0
        # recorrer cada dimensión de los puntos
        for i in range(len(punto_a)):
            # calcular la diferencia en la dimensión i
            diferencia = punto_a[i] - punto_b[i]
            # elevar al cuadrado y acumular
            suma_cuadrados += diferencia ** 2
        # retornar la raíz cuadrada de la suma de cuadrados
        return math.sqrt(suma_cuadrados)

    def clasificar(self, punto_nuevo: list[float]) -> str:
        """
        Clasifica un punto nuevo usando voto mayoritario de los k vecinos más cercanos.
        1. Calcula la distancia del punto nuevo a todos los puntos de entrenamiento
        2. Ordena de menor a mayor distancia
        3. Toma los 3 más cercanos (los k vecinos)
        4. Cuenta cuántos son de cada clase
        5. Gana el de mayoria
        """

        # recorder que datos de entraniemto es una arreglo de areglos
        # calcular la distancia del punto nuevo a cada punto de entrenamiento

        # guarda la distancia entre la clase y los datos de entrenamiento
        distancias: list[tuple[float, str]] = []
        for i in range(len(self.datos_entrenamiento)):
            # calcular la distancia entre el punto nuevo y el punto de entrenamiento i
            # las distancia entre los puntos de un punto al predecir y los caracteristicas
            dist = self._distancia_euclidiana(punto_nuevo, self.datos_entrenamiento[i])
            # guardar la distancia junto con la clase del punto de entrenamiento
            # guardamos la distancia con su clase para saber cuales son las clases mas cercanas
            distancias.append((dist, self.clases_entrenamiento[i]))
        # ordenar la lista de distancias de menor a mayor (por el primer elemento de la tupla)
        distancias.sort(key=lambda x: x[0]) # recordar el primer elemento de la tupla es la distancia
        # extraer las clases de los k vecinos más cercanos (los primeros k de la lista ordenada)
        # extraer los strings (nombre de la clase ) las tuplas, despues ser ordenadas
        k_vecinos = [clase for _, clase in distancias[:self.k]]
        # contar cuántas veces aparece cada clase entre los k vecinos
        conteo = Counter(k_vecinos)
        # retornar la clase que más veces aparece (voto mayoritario)
        # regresa una lista con las tuplas de clase con mayor frecuencia
        # de la lista toma la tupla,
        # de la tupla toma la clase con mayor frecuencia de los mas cercanos
        return conteo.most_common(1)[0][0]

    def evaluar_restitucion(self) -> float:
        """Evalúa el modelo usando el método de restitución (mismos datos para entrenar y evaluar)."""
        # contador de predicciones correctas
        aciertos = 0
        # recorrer cada punto del conjunto de entrenamiento
        for i in range(len(self.datos_entrenamiento)):
            # clasificar el punto i usando todos los datos de entrenamiento
            prediccion = self.clasificar(self.datos_entrenamiento[i])
            # comparar la predicción con la clase real del punto
            if prediccion == self.clases_entrenamiento[i]:
                # si coinciden, incrementar el contador de aciertos
                aciertos += 1
        # calcular el porcentaje de aciertos sobre el total de muestras
        precision = aciertos / len(self.datos_entrenamiento) * 100
        # retornar la precisión como porcentaje
        return precision
