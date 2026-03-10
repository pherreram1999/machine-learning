import polars as pl
from typing import List
import numpy as np

from NaiveBayes import NaiveBayes
from PreditionCollection import PreditionCollection


class NaiveBayesDiscreto(NaiveBayes):

    def entrenar(self):
        # obtiene las frecuencias de cada caracteristicas dada que si pasa Yr
        # al hacer unpivot reducirmos las columnas a Yr | Variable | Value
        # es decir cada columna ahora es trasladada a la columna Variable
        # y el valor correspondiente de la fila se coloca en Value
        # es facilita agruparls por Yr, variable, y su valor para poder contar frecuencia
        # similar a sql con los groups
        # Es como aplastar los datos
        self._frecuencias_Xi = self._data.unpivot(index=self.columnYr).group_by(
            [self.columnYr, "variable", "value"]).len()

        # sacamos el numero de categorais por caracteristica para el suavizado de laplace
        self._num_valores = {}
        for col in self.columnas[:-1]:
            self._num_valores[col] = self._data[col].n_unique()

    def predecir(self, input: List):

        probabilidades_yr = {}
        for etiqueta, frecuencia_yr in self._frecuencias_Yr.rows():

            # la probabildad de esta etiqueta o clase empiza con la apriori
            probabilidades_yr[etiqueta] = frecuencia_yr / self.num_muestras

            for i, valor_buscado in enumerate(input):
                X_name = self.columnas[i]

                # de las combinaciones de frecuencias encontramos las que concidan con la entrada
                # para obtener P(Xi| Yr)
                res = self._frecuencias_Xi.filter(
                    (pl.col("variable") == X_name) &
                    (pl.col("value") == str(valor_buscado)) &
                    (pl.col(self.columnYr) == etiqueta)
                )

                frecuencia_xi = res['len'][0] if len(res) > 0 else 0

                # se hace los productos de las probabilidades (con Laplace smoothing)
                k = self._num_valores[X_name]
                probabilidades_yr[etiqueta] *= (frecuencia_xi + 1) / (frecuencia_yr + k)

        return PreditionCollection(NaiveBayes.normalizar(probabilidades_yr))

