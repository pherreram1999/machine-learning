from pydoc import apropos

import polars as pl

class NaiveBayes:

    @classmethod
    def discrete(cls,path = "data.csv"):
        cls.__data = pl.read_csv(path, columns=["Clima","Temperatura","Humedad","Viento","Juego"])
        # obtenemos la frencuencia de los valores de las columnas
        # de nuestra etiqueta
        frecuencia = cls.__data['Juego'].value_counts()

        cls.rows_size, _ = cls.__data.shape


        apriori = {} # lo guardamos en un diccionario

        for key,val in frecuencia.rows():
            apriori[key] = val

        cls.__apriori = apriori

        return cls()

    def Ask(self, input):
        _, cols = self.__data.shape

        if len(input) != cols - 1:
            raise Exception(f'El input debe ser un arreglo de {cols -1}')

        # segun la entrada buscamos la probabilidad de cada uno de sus elemtnos
        # segun lso valores apriori

        # buscamos la frencia por cada de los valores a priori
        # se crea por cada etiqueta, un arreglo de la frecuencia del input dado
        frecuencia_dada = {}
        for etiqueta in self.__apriori.keys():
            frecuencia_dada[etiqueta] = [0] * ( cols - 1)


        for etiqueta, frecuencia in self.__apriori.items():
            # buscamos la probabilidad de cada una de las entradas
            # por cada etiqueta, buscamos la probabildiad P(X|etiqueta)

            for Xi in range(len(input)):
                for row in self.__data.rows():
                    # si la etiqueta actual es igual a Yobt,
                    if etiqueta == row[-1] and row[Xi] == input[Xi]:
                        frecuencia_dada[etiqueta][Xi] += 1
                        pass
                pass
        pass

        # Nota: al estar basandose solo en las letras, se suscriben las probilidade, quiza baserse en una combinacion
        # por ultimo queda dividor para aplicar frecuencias

        probabilidades = frecuencia_dada.copy()





