import numpy as np
from Perceptron import Perceptron

"""
Adaline SGD / Descenso de Gradiente Estocástico
1. El producto punto calcula la entrada neta (z) para UNA sola fila a la vez.
2. Durante el ENTRENAMIENTO, la activación es LINEAL. Evaluamos el error usando 'z'.
3. Al inicio de cada época, se MEZCLAN (shuffle) los datos para evitar ciclos de aprendizaje estancados.
4. Se actualizan los pesos de forma INSTANTÁNEA (fila por fila) multiplicando el error por sus características específicas.
5. El costo global de la época se calcula promediando los errores individuales para revisar la convergencia (tolerancia).
"""
class PerceptronSGD(Perceptron):

    def __init__(self, epochs: int, eta: float = 0.01,
                 random_seed: float = 1, tol: float = 1e-6):
        # Heredamos variables y el scaler de la clase base
        super().__init__(epochs, eta, random_seed)

        self._tol = tol
        self._costos: list = []
        # Guardamos el generador en una variable de clase para usarlo en el shuffle
        self._rgen = np.random.RandomState(self._random_seed)

    def _shuffle(self, X, y):
        """
        Mezcla los datos de entrenamiento para el comportamiento estocástico.
        Genera índices aleatorios y reordena las matrices X y y.
        """
        r = self._rgen.permutation(len(y))
        return X[r], y[r]

    def entrenar(self, X, y):
        X_scaled = self._scaler.fit_transform(X)

        # Adaline funciona mejor con targets {-1, +1}
        # mapeamos los valores para mejor toma de decisiones
        y_signed = np.where(np.asarray(y) == 1, 1, -1)

        # Inicializamos los pesos de manera aleatoria
        self._w = self._rgen.normal(loc=0.0, scale=0.01, size=1 + X_scaled.shape[1])
        self._costos = []

        for _ in range(self._epochs):
            # 1. EL TRUCO SGD: Mezclamos los datos al iniciar la época
            # esto para no tener un sesgo que afecte el entrenamiento
            X_shuffled, y_shuffled = self._shuffle(X_scaled, y_signed)

            costo_epoca = []

            # 2. CICLO FILA POR FILA (Sobre la marcha)
            for xi, target in zip(X_shuffled, y_shuffled):
                # ENTRADA NETA (z): Usamos el método heredado para una sola fila
                predict = self.dotProduct(xi)

                # ERROR CONTINUO: Diferencia entre lo real y la entrada neta de esta fila
                error = target - predict

                # ACTUALIZACIÓN INMEDIATA: Aplicamos la regla delta directo a los pesos
                # Ya no dividimos entre 'n' porque el ajuste es para una sola persona
                self._w[1:] += self._eta * error * xi
                self._w[0] += self._eta * error

                # FUNCIÓN DE COSTO (SSE) INDIVIDUAL: 1/2 * error^2
                costo_epoca.append(0.5 * (error ** 2))

            # PROMEDIO DEL COSTO: Sumamos los costos de toda la fila y promediamos
            avg_cost = sum(costo_epoca) / len(y_shuffled)
            self._costos.append(avg_cost)

            # DIAGNÓSTICO DE CONVERGENCIA (Parada Temprana)
            # Evalúa con el costo promediado de la época
            if len(self._costos) > 1 and \
                    abs(self._costos[-2] - self._costos[-1]) < self._tol:
                break

    @property
    def costos(self):
        return self._costos