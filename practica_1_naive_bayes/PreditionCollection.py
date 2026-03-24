from typing import Dict
from Prediction import Prediction


class PreditionCollection:

    def __init__(self, probabilidades: Dict[str, float]):
        self.__probabilidades = []
        for etiqueta, prob in probabilidades.items():
            self.__probabilidades.append(Prediction(etiqueta, prob))

    def __str__(self):
        res = ''
        for prob in self.__probabilidades:
            res += str(prob) + '\n'

        m = self.max()

        res += 'max: ' + str(m) + '\n'
        return res

    def max(self):
        m = self.__probabilidades[0]
        for p in self.__probabilidades:
            if p.probabilidad > m.probabilidad:
                m = p
        return m