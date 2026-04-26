import numpy as np
from Perceptron import Perceptron


class PerceptronGD(Perceptron):
    """
    Variante del perceptron usando descenso gradiente heuristico (Adaline).
    Diferencias clave respecto al perceptron clasico:
      - el error se calcula con la salida lineal (net input), no con la
        prediccion binaria del escalon
      - actualizacion batch: se promedia el gradiente sobre todas las muestras
        por epoca, en vez de actualizar muestra por muestra
      - se registra el costo SSE por epoca para diagnostico de convergencia
    """

    def __init__(self, epochs: int, eta: float = 0.01,
                 random_seed: float = 1, tol: float = 1e-6):
        super().__init__(epochs, eta, random_seed)
        # tolerancia para parar si la mejora del costo es despreciable
        self._tol = tol
        self._costos: list = []

    def entrenar(self, X, y):
        # misma normalizacion que la clase padre
        X_scaled = self._scaler.fit_transform(X)

        # Adaline funciona mucho mejor con targets {-1, +1} en vez de {0, 1}:
        # el umbral de decision es net_input > 0, asi el SSE empuja la salida
        # lineal hacia el lado correcto del 0 en vez de hacia 0.5
        y_signed = np.where(np.asarray(y) == 1, 1, -1)

        rgen = np.random.RandomState(self._random_seed)

        # incializamos los pesos de manera aleatoria
        self._w = rgen.normal(loc=0.0, scale=0.01, size=1 + X_scaled.shape[1])
        self._costos = []

        n = len(y_signed)

        for _ in range(self._epochs):
            # salida lineal phi(z) = z, sin aplicar escalon
            net_input = self.rule(X_scaled)

            # error continuo: y - phi(z), con y en {-1, +1}
            errores = y_signed - net_input

            # gradiente del SSE = -X^T * errores
            # actualizamos pesos restando el gradiente (osea sumando el error)
            # promedio por N para que eta no dependa del tamaño del batch
            self._w[1:] += self._eta * X_scaled.T.dot(errores) / n
            self._w[0] += self._eta * errores.sum() / n

            # SSE: J(w) = 1/2 * sum((y - phi(z))^2)
            costo = (errores ** 2).sum() / 2.0
            self._costos.append(costo)

            # si el costo dejo de bajar de manera apreciable, paramos
            if len(self._costos) > 1 and \
                    abs(self._costos[-2] - self._costos[-1]) < self._tol:
                break

    @property
    def costos(self):
        return self._costos