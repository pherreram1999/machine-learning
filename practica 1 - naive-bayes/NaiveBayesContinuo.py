import polars as pl
from typing import List
import numpy as np

from NaiveBayes import NaiveBayes
from PreditionCollection import PreditionCollection


class NaiveBayesContinuo(NaiveBayes):

    columnas = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]

    def __init__(self, path="Iris.csv"):
        NaiveBayes.__init__(self, path)

    def entrenar(self):
        Yd_column = self.columnas[-1]

        # consigue las medias de cada uno de las caracteristicas por especie
        self.medias = self._mapping(self._data.group_by(Yd_column).mean())
        # consigue las varianzas por cada uno de las caracteristicas por especie
        # el agregate es para realizar operaciiones sobre agrupaciones
        # se hace asi dado que la varianza no se puede aplicar directo a un grupo
        # no contamos lo de la etiqueta
        self.varianzas = self._mapping(self._data.group_by(Yd_column).agg(pl.exclude(Yd_column).var()))

    def _mapping(self, tabla):
        dic = {}
        for row in tabla.rows():
            dic[row[0]] = row[1:]
        return dic

    def gaussiana(self, X, media, varianza):
        return (1 / np.sqrt(2 * np.pi * varianza)) * np.exp(-((X - media) ** 2) / (2 * varianza))

    def predecir(self, input: List):
        probabilidades_por_especie = {}
        # se saca la probabilidad por cada una de las especiaes
        for etiqueta, frecuencia_yr in self._frecuencias_Yr.rows():
            # varianzas y media por especie
            var = self.varianzas[etiqueta]
            media = self.medias[etiqueta]

            # lo incializamos en uno para mantener la primera probaliidad
            # probabilidad apriori
            probabilidades_por_especie[etiqueta] = frecuencia_yr / self.num_muestras
            # recorremos cada caracteriticas
            # nos basamos en orden de entrada
            for i, valor_buscado in enumerate(input):
                # varianza y media por caracteristica
                v = var[i]
                m = media[i]
                probabilidades_por_especie[etiqueta] *= self.gaussiana(valor_buscado, m, v)

        return PreditionCollection(NaiveBayes.normalizar(probabilidades_por_especie))