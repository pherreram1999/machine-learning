class ResultadoRestitucion:

    def __init__(self, acertadas: int, total: int):
        self._acertadas = acertadas
        self._total = total

    @property
    def porcentaje(self) -> float:
        return (self._acertadas / self._total) * 100

    def __str__(self):
        return (
            f'Comprobacion por restitucion\n'
            f'  Acertadas : {self._acertadas} / {self._total}\n'
            f'  Porcentaje: {self.porcentaje:.2f}%'
        )