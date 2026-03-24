from knn import KNN  # importar la clase base KNN para heredar de ella


class KNNPonderado(KNN):
    """K-NN con pesos: los vecinos más cercanos tienen mayor influencia en la clasificación.

    En vez de un voto simple (1 voto por vecino), cada vecino vota con peso = 1/distancia.
    Así, un vecino muy cercano influye más que uno lejano.
    """

    def clasificar(self, punto_nuevo: list[float]) -> str:
        """Clasifica un punto nuevo usando voto ponderado por distancia inversa."""
        # calcular la distancia del punto nuevo a cada punto de entrenamiento
        distancias: list[tuple[float, str]] = []
        for i in range(len(self.datos_entrenamiento)):
            # calcular la distancia entre el punto nuevo y el punto de entrenamiento i
            dist = self._distancia_euclidiana(punto_nuevo, self.datos_entrenamiento[i])
            # guardar la distancia junto con la clase del punto de entrenamiento
            distancias.append((dist, self.clases_entrenamiento[i]))
        # ordenar la lista de distancias de menor a mayor
        distancias.sort(key=lambda x: x[0])
        # tomar solo los k vecinos más cercanos
        k_vecinos = distancias[:self.k]
        # verificar si algún vecino tiene distancia 0 (es el mismo punto)
        for dist, clase in k_vecinos:
            if dist == 0.0:
                # si la distancia es 0, el punto es idéntico: retornar su clase directamente
                return clase
        # acumular los pesos por clase usando un diccionario
        pesos_por_clase: dict[str, float] = {}
        for dist, clase in k_vecinos:
            # calcular el peso como el inverso de la distancia (más cerca = más peso)
            peso = 1.0 / dist
            # si la clase ya está en el diccionario, sumar el peso; si no, inicializarla
            if clase in pesos_por_clase:
                pesos_por_clase[clase] += peso  # acumular peso para esta clase
            else:
                pesos_por_clase[clase] = peso  # primera vez que vemos esta clase
        # encontrar la clase con mayor peso acumulado
        clase_ganadora = max(pesos_por_clase, key=lambda c: pesos_por_clase[c])
        # retornar la clase con mayor peso total
        return clase_ganadora
