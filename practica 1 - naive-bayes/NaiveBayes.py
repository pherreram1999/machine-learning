import polars as pl
from typing import Dict, List
import numpy as np

from PreditionCollection import PreditionCollection
from ResultadoRestitucion import ResultadoRestitucion


class NaiveBayes:

    columnas = ["Clima", "Temperatura", "Humedad", "Viento", "Juego"]

    @classmethod
    def entrenar(self):
        pass

    def __init__(self, path="data.csv"):
        """"Carga los datos de la fuente"""

        self._data = pl.read_csv(path, columns=self.columnas)
        # obtenemos la frencuencia de los valores de las columnas
        # de nuestra etiqueta

        self.columnYr = self.columnas[-1]

        # agrupamos por columna Yr para contar su freucencia
        self._frecuencias_Yr = self._data.group_by(self.columnYr).len()
        # aqui se manda a llamar el entranmiento para discreto o continuo
        self.entrenar()



        self.num_muestras, _ = self._data.shape


    @staticmethod
    def normalizar(probabilidades: Dict[str, float]) -> Dict[str, float]:
        listaProbabilades = list(probabilidades.values())
        sum = np.sum(listaProbabilades)
        normalizado = {}
        for etiqueta, prob in probabilidades.items():
            normalizado[etiqueta] = (prob / sum) * 100
        return normalizado

    @staticmethod
    def print(probabilidades: Dict[str, float]):
        for etiqueta, prob in probabilidades.items():
            print(f'Etiqueta: {etiqueta} | Probabilidade: {prob}.2f')

    @classmethod
    def predecir(self, input):
        """  pide el un arreglo con los valores predecir """
        pass

    @staticmethod
    def restituir(fila, archivo):
        filas = pl.read_csv(archivo)
        rest = list(filas.row(fila)[1:5])
        return rest

    def comprobar_por_restitucion(self) -> ResultadoRestitucion:
        acertadas = 0
        for row in self._data.rows():
            Xi = list(row[:-1])
            etiqueta_real = row[-1]
            prediccion = self.predecir(Xi)
            if prediccion.max().etiqueta == etiqueta_real:
                acertadas += 1
        return ResultadoRestitucion(acertadas, self.num_muestras)