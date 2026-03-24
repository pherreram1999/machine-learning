class Prediction:

    def __init__(self, etiqueta: str, probabilidad: float):
        self._etiqueta = etiqueta
        self._probabilidad = probabilidad

    def __str__(self):
        return f'Etiqueta: {self._etiqueta} - Probabilidad: {self._probabilidad:.2f}'

    @property
    def etiqueta(self):
        return self._etiqueta

    @property
    def probabilidad(self):
        return self._probabilidad