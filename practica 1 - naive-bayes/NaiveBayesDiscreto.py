import polars as pl
from typing import List
import numpy as np

from NaiveBayes import NaiveBayes
from PreditionCollection import PreditionCollection


class NaiveBayesDiscreto(NaiveBayes):

    @staticmethod
    def normal(probabilidad, probas):
        return probabilidad / np.sum(probas)

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

                # se hace los productos de las probabilidades
                probabilidades_yr[etiqueta] *= frecuencia_xi / frecuencia_yr

        return PreditionCollection(NaiveBayes.normalizar(probabilidades_yr))

    def ___predecir(self, input: List):
        _, cols = self._data.shape

        input_length = len(input)

        if input_length != cols - 1:
            raise Exception(f'El input debe ser un arreglo de {cols - 1}')

        # segun la entrada buscamos la probabilidad de cada uno de sus elemtnos
        # segun lso valores apriori

        # buscamos la frencia por cada de los valores la etiquetas
        # se crea por cada etiqueta, un arreglo de la frecuencia del input dado
        frecunciaXn = {}  # gyardadomos las frecuencias
        for etiqueta in self._Yr.keys():
            frecunciaXn[etiqueta] = [0] * (cols - 1)

        # donde guardaron las probabilidades de Yoby
        Yobt = {}

        for etiqueta, _ in self._Yr.items():
            # buscamos la probabilidad de cada una de las entradas
            # por cada etiqueta, buscamos la probabildiad P(X|etiqueta)

            for Xi in range(input_length):
                for row in self._data.rows():
                    # si la etiqueta actual es igual a Yr,
                    if etiqueta == row[-1] and row[Xi] == input[Xi]:
                        frecunciaXn[etiqueta][Xi] += 1

            # una vez que se conto las apereciones dadas
            frecunciaYr = self._Yr[etiqueta]
            # se calcula la probilidad dad su frencuencia
            probabilida_Yr = frecunciaYr / self.num_muestras

            for Xi in range(input_length):
                probabilida_Yr *= frecunciaXn[etiqueta][Xi] / frecunciaYr

            # una vez que se tiene las probabilidades
            Yobt[etiqueta] = probabilida_Yr

        # Nota: al estar basandose solo en las letras, se suscriben las probilidade, quiza baserse en una combinacion
        # por ultimo queda dividor para aplicar frecuencias

        return Yobt