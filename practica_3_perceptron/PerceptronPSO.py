import numpy as np
from Perceptron import Perceptron


class PerceptronPSO(Perceptron):
    """
    Variante del perceptron entrenado con Particle Swarm Optimization.
    No usa gradientes: cada particula es un vector de pesos candidato que
    se mueve por el espacio de busqueda guiada por:
      - su mejor posicion personal historica (pbest)
      - la mejor posicion global del enjambre (gbest)
    Funcion de fitness: numero de errores de clasificacion (a minimizar).
    """

    def __init__(self, epochs: int, eta: float = 0.1, random_seed: float = 1,
                 n_particulas: int = 30, w_inercia: float = 0.7,
                 c1: float = 1.5, c2: float = 1.5):
        # eta no se usa aqui (PSO no tiene learning rate clasico) pero se
        # acepta por compatibilidad con la firma de la clase padre
        super().__init__(epochs, eta, random_seed)
        self._n_particulas = n_particulas
        self._w_inercia = w_inercia  # peso de inercia
        self._c1 = c1  # coeficiente cognitivo (atraccion al pbest)
        self._c2 = c2  # coeficiente social (atraccion al gbest)
        self._historial_fitness: list = []

    def _fitness(self, w, X, y):
        # numero de muestras mal clasificadas (a minimizar)
        z = np.dot(X, w[1:]) + w[0]
        pred = np.where(z > 0, 1, 0)
        return int((pred != y).sum())

    def entrenar(self, X, y):
        X_scaled = self._scaler.fit_transform(X)
        y = np.asarray(y)

        rgen = np.random.RandomState(self._random_seed)
        dim = 1 + X_scaled.shape[1]

        # inicializamos enjambre: posiciones y velocidades aleatorias
        particulas = rgen.normal(loc=0.0, scale=0.5,
                                 size=(self._n_particulas, dim))
        velocidades = rgen.normal(loc=0.0, scale=0.1,
                                  size=(self._n_particulas, dim))

        # mejor posicion personal de cada particula
        pbest = particulas.copy()
        pbest_fit = np.array([self._fitness(p, X_scaled, y)
                              for p in particulas])

        # mejor global del enjambre
        idx = pbest_fit.argmin()
        gbest = pbest[idx].copy()
        gbest_fit = pbest_fit[idx]

        self._historial_fitness = [gbest_fit]

        for _ in range(self._epochs):
            # componentes estocasticos por particula y dimension
            r1 = rgen.random((self._n_particulas, dim))
            r2 = rgen.random((self._n_particulas, dim))

            # actualizamos velocidades: inercia + cognitivo + social
            velocidades = (self._w_inercia * velocidades
                           + self._c1 * r1 * (pbest - particulas)
                           + self._c2 * r2 * (gbest - particulas))

            # actualizamos posiciones
            particulas += velocidades

            # evaluamos fitness y refrescamos pbest/gbest
            for i in range(self._n_particulas):
                f = self._fitness(particulas[i], X_scaled, y)
                if f < pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest[i] = particulas[i].copy()
                    if f < gbest_fit:
                        gbest_fit = f
                        gbest = particulas[i].copy()

            self._historial_fitness.append(gbest_fit)

            # convergencia perfecta, ya no hace falta seguir
            if gbest_fit == 0:
                break

        # los pesos finales son los del mejor global encontrado
        self._w = gbest

    @property
    def historial_fitness(self):
        return self._historial_fitness