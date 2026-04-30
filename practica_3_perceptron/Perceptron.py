import numpy as np
from sklearn.preprocessing import StandardScaler


"""
Perceptron clasico
1. El producto punto es el calculo de la z = x1w1 +  x2w2 + ... + xnwn + b
2. la funcion escalon es la activacion que adentro llama a (1)
3. durante el entranimiento se incializan los pesos aleatoriamente
4. se hacen "predicciones" que es llamar a (2), medimos que tan lejos esta del error con la regla delta
5. aplicamos actualizacion de los pesos dado: wj = wj + Delta(Wj) donde la delta es la diferencia de lo obtenido respecto a lo real
"""
class Perceptron:

    def __init__(self, epochs: int, eta: float = 0.1, random_seed: float = 1):
        self._eta = eta # learning rate
        self._epochs = epochs
        self._random_seed = random_seed
        self._scaler = StandardScaler()
        self._w: np.ndarray = np.array([])

    def dotProduct(self, X):
        """
        :param X:
        :return:
        """
        # recordar w[0] es el bias
        # esto es la operacion
        # $$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$
        return np.dot(X, self._w[1:]) + self._w[0]

    def _escalon(self, X):
        """
        Aplica funcion escalon a salida de regla delta
        :param X:
        :return:
        """
        return np.where(self.dotProduct(X) > 0, 1, 0)

    def predecir(self, X):
        """
        Retoran un arreglo ya con al funcion de activacion
        :param X:
        :return:
        """
        # mapea los valores donde si es mayor a 0 es 1 de lo contrario es 0
        # que es la funcion escalon para este caso
        X_scaled = self._scaler.transform(X)
        return self._escalon(X_scaled)

    def entrenar(self, X, y):
        # normalizamos con StandardScaler (media=0, std=1) para que ninguna
        # feature domine los pesos por tener mayor escala
        X_scaled = self._scaler.fit_transform(X)

        rgen = np.random.RandomState(self._random_seed)

        # incializamos los pesos de manera aleatoria
        self._w = rgen.normal(loc=0.0, scale=0.01, size=1 + X_scaled.shape[1])

        for _ in range(self._epochs):
            for xi, target in zip(X_scaled, y):
                # aplica la regla, que es delta
                # aplica mape aplicando escalon a la prediccion
                predicted = self._escalon(xi)

                # si diferencia es 0, prediccion correcta, no hay que actualizar pesos
                if target == predicted:
                    continue

                # aplicamos la regla de delta
                # quien se encarga de aplicar los pesos
                update = self._eta * (target - predicted)
                # actualizamos los pesos
                # Actualizamos los pesos de las características (w1, w2, ..., wn)
                # donde cada peso es multiplicado por una caracteritica
                self._w[1:] += update * xi

                # Actualizamos el peso del bias (w0)
                self._w[0] += update