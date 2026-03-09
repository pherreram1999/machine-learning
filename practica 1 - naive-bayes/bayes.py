import polars as pl
from typing import Tuple, Dict, List, Set
import numpy as np

class Prediction:

    def __init__(self,etiqueta: str, probabilidad: float):
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

class PreditionCollection:


    def __init__(self,probabilidades: Dict[str,float]):
        self.__probabilidades = []
        for etiqueta, prob in probabilidades.items():
            self.__probabilidades.append(Prediction(etiqueta, prob))


    def __str__(self):
        res = ''
        for prob in self.__probabilidades:
            res += str(prob) + '\n'
        return res

    def max(self):
        m = self.__probabilidades[0]
        for p in self.__probabilidades:
            if p.probabilidad > m.probabilidad:
                m = p
        return m



class NaiveBayes:

    columnas = ["Clima", "Temperatura", "Humedad", "Viento", "Juego"]

    @classmethod
    def entrenar(self):
        pass

    def __init__(self,path = "data.csv"):
        """"Carga los datos de la fuente"""

        self._data = pl.read_csv(path, columns= self.columnas)
        # obtenemos la frencuencia de los valores de las columnas
        # de nuestra etiqueta

        self.columnYr = self.columnas[-1]

        # agrupamos por columna Yr para contar su freucencia
        self._frecuencias_Yr = self._data.group_by(self.columnYr).len()
        # aqui se manda a llamar el entranmiento para discreto o continuo
        self.entrenar()


        frecuencia = self._data[self.columnas[-1]].value_counts()

        self.num_muestras, _ = self._data.shape

        self._clases = self._data[self.columnYr].unique()

        Yr = {}  # lo guardamos en un diccionario

        for key, val in frecuencia.rows():
            Yr[key] = val

        self._Yr = Yr
        pass

    @staticmethod
    def normalizar(probabilidades: Dict[str, float]) -> Dict[str, float]:
        listaProbabilades = list(probabilidades.values())
        sum = np.sum(listaProbabilades)
        normalizado = {}
        for etiqueta, prob in probabilidades.items():
            normalizado[etiqueta] = (prob / sum) * 100
        return normalizado

    @staticmethod
    def print(probabilidades: Dict[str,float]):
        for etiqueta, prob in probabilidades.items():
            print(f'Etiqueta: {etiqueta} | Probabilidade: {prob}.2f')


    @classmethod
    def predecir(self, input):
        """  pide el un arreglo con los valores predecir """
        pass

    @staticmethod
    def restituir(fila,archivo):
        filas = pl.read_csv(archivo)
        rest = list(filas.row(fila)[1:5])
        return rest

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
                pass
        pass

        return PreditionCollection(NaiveBayes.normalizar(probabilidades_yr))

    def ___predecir(self, input: List) -> Dict[str, float]:
        _, cols = self._data.shape

        input_length = len(input)



        if input_length != cols - 1:
            raise Exception(f'El input debe ser un arreglo de {cols -1}')


        # segun la entrada buscamos la probabilidad de cada uno de sus elemtnos
        # segun lso valores apriori

        # buscamos la frencia por cada de los valores la etiquetas
        # se crea por cada etiqueta, un arreglo de la frecuencia del input dado
        frecunciaXn = {} # gyardadomos las frecuencias
        for etiqueta in self._Yr.keys():
            frecunciaXn[etiqueta] = [0] * ( cols - 1)

        # donde guardaron las probabilidades de Yoby
        Yobt = {}

        for etiqueta,_ in self._Yr.items():
            # buscamos la probabilidad de cada una de las entradas
            # por cada etiqueta, buscamos la probabildiad P(X|etiqueta)

            for Xi in range(input_length):
                for row in self._data.rows():
                    # si la etiqueta actual es igual a Yr,
                    if etiqueta == row[-1] and row[Xi] == input[Xi]:
                        frecunciaXn[etiqueta][Xi] += 1
                        pass
                pass

            # una vez que se conto las apereciones dadas
            frecunciaYr = self._Yr[etiqueta]
            # se calcula la probilidad dad su frencuencia
            probabilida_Yr = frecunciaYr / self.num_muestras

            for Xi in range(input_length):
                probabilida_Yr *= frecunciaXn[etiqueta][Xi] / frecunciaYr
                pass

            # una vez que se tiene las probabilidades
            Yobt[etiqueta] = probabilida_Yr
        pass
        # Nota: al estar basandose solo en las letras, se suscriben las probilidade, quiza baserse en una combinacion
        # por ultimo queda dividor para aplicar frecuencias

        return Yobt



class NaiveBayesContinuo(NaiveBayes):

    columnas = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm","Species"]

    def entrenar(self):
        Yd_column = self.columnas[-1]

        # consigue las medias de cada uno de las caracteristicas por especie
        self.medias = self._mapping(self._data.group_by(Yd_column).mean())
        # consigue las varianzas por cada uno de las caracteristicas por especie
        # el agregate es para realizar operaciiones sobre agrupaciones
        # se hace asi dado que la varianza no se puede aplicar directo a un grupo
        # no contamos lo de la etiqueta
        self.varianzas = self._mapping(self._data.group_by(Yd_column).agg(pl.exclude(Yd_column).var()))

    def __init__(self, path="Iris.csv"):
        NaiveBayes.__init__(self, path)
        pass


    def _mapping(self, tabla):
        dic = {}
        for row in tabla.rows():
            dic[row[0]] = row[1:]
        return dic

    def gaussiana(self,X,media,varianza):
        return (1 / np.sqrt(2 * np.pi * varianza)) * np.exp(- (( (X-media) ** 2) /  (2*varianza)))

    def predecir(self, input: List):
        probabilidades_por_especie = {}
        # se saca la probabilidad por cada una de las especiaes
        for etiqueta, frecuencia_yr in self._frecuencias_Yr.rows():
            # varianzas y media por especie
            var = self.varianzas[etiqueta]
            media = self.medias[etiqueta]

            # lo incializamos en uno para mantener la primera probaliidad
            # probabilidad apriori
            probabilidades_por_especie[etiqueta] = self._Yr[etiqueta] / self.num_muestras
            # recorremos cada caracteriticas
            # nos basamos en orden de entrada
            for Xi in range(len(input)):
                # varianza y media por caracteristica
                v = var[Xi]
                m = media[Xi]
                probabilidades_por_especie[etiqueta] *= self.gaussiana(input[Xi],m,v)
                pass
        pass
        return PreditionCollection(NaiveBayes.normalizar(probabilidades_por_especie))





