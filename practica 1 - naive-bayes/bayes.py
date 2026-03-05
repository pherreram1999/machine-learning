from pydoc import apropos

import polars as pl


class NaiveBayes:

    columnas = ["Clima", "Temperatura", "Humedad", "Viento", "Juego"]

    def __init__(self,path = "data.csv"):
        """"Carga los datos de la fuente"""

        self._data = pl.read_csv(path, columns= self.columnas)
        # obtenemos la frencuencia de los valores de las columnas
        # de nuestra etiqueta
        frecuencia = self._data['Juego'].value_counts()

        self.num_muestras, _ = self._data.shape

        Yr = {}  # lo guardamos en un diccionario

        for key, val in frecuencia.rows():
            Yr[key] = val

        self._Yr = Yr
        pass


    @classmethod
    def ask(self, input):
        """  pide el un arreglo con los valores predecir """
        pass


class NaiveBayesDiscreto(NaiveBayes):


    def ask(self, input):
        _, cols = self._data.shape

        input_length = len(input)

        if input_length != cols - 1:
            raise Exception(f'El input debe ser un arreglo de {cols -1}')

        # segun la entrada buscamos la probabilidad de cada uno de sus elemtnos
        # segun lso valores apriori

        # buscamos la frencia por cada de los valores a priori
        # se crea por cada etiqueta, un arreglo de la frecuencia del input dado
        frecunciaXn = {}
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
            probabilida_Yr = self._Yr[etiqueta] / self.num_muestras

            for Xi in range(input_length):
                probabilida_Yr *= frecunciaXn[etiqueta][Xi] / frecunciaYr
                pass

            # una vez que se tiene las probabilidades


            Yobt[etiqueta] = probabilida_Yr

        pass

        # Nota: al estar basandose solo en las letras, se suscriben las probilidade, quiza baserse en una combinacion
        # por ultimo queda dividor para aplicar frecuencias

        return Yobt





