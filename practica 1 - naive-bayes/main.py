import polars as pl

from bayes import NaiveBayesDiscreto,NaiveBayesContinuo,NaiveBayes


def main():

    # --- Modelo discreto (datos.csv) ---
    discreto = NaiveBayesDiscreto()
    b = 0  # contador de predicciones correctas
    arch = pl.read_csv('datos.csv')

    for i in range(len(arch)):
        # se retira la fila i del dataset para usarla como muestra de prueba
        filas = discreto.restituir(i,'datos.csv')
        # se predice la clase para la muestra
        prob = discreto.predecir(filas)
        # se normalizan las probabilidades a porcentaje
        norm = NaiveBayesDiscreto.normalizar(prob)
        print(norm)

        # se elige la clase con mayor probabilidad
        if list(norm.values())[0] > list(norm.values())[1]:
            a = 'N'
        else:
            a = 'S'

        # se compara con el valor real de la fila
        if a == arch[i,5]:
            b += 1

    print("\nFiabilidad del modelo: %",b)

    # --- Modelo continuo (iris.csv) ---
    continuo = NaiveBayesContinuo()

    # predicción puntual de ejemplo
    probabilidades = continuo.predecir([6.1,2.9,4.7,1.4])
    print("\n")
    NaiveBayesContinuo.print(NaiveBayesContinuo.normalizar(probabilidades))

    c = 0  # contador de predicciones correctas
    arch2 = pl.read_csv('iris.csv')

    for i in range(len(arch2)):
        # se retira la fila i del dataset para usarla como muestra de prueba
        filas2 = continuo.restituir(i,'iris.csv')
        prob2 = continuo.predecir(filas2)
        norm2 = NaiveBayesContinuo.normalizar(prob2)
        print(norm2)

        # se elige la especie con mayor probabilidad
        if list(norm2.values())[0] > list(norm2.values())[1] and list(norm2.values())[0] > list(norm2.values())[2]:
            d = "Iris-setosa"
        elif list(norm2.values())[1] > list(norm2.values())[0] and list(norm2.values())[1] > list(norm2.values())[2]:
            d = "Iris-versicolor"
        else:
            d = "Iris-virginica"

        # se compara con el valor real de la fila
        if d == arch2[i,5]:
            c += 1

    print(c)
    pass


if __name__ == '__main__':
    main()