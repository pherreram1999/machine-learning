import csv
import pandas as pd

from bayes import NaiveBayesDiscreto,NaiveBayesContinuo,NaiveBayes


def main():

    discreto = NaiveBayesDiscreto()
    b = 0
    arch = pd.read_csv('datos.csv')
    for i in range(len(arch)):
        filas = discreto.restituir(i,'datos.csv')
        prob = discreto.predecir(filas)
        norm = NaiveBayesDiscreto.normalizar(prob)
        print(norm)
        if list(norm.values())[0] > list(norm.values())[1]:
            a = 'N'
        else:
            a = 'S'

        if a == arch.iloc[i,5]:
            b += 1

    print("\nFiabilidad del modelo: %",b)
    continuo = NaiveBayesContinuo()
    probabilidades = continuo.predecir([6.1,2.9,4.7,1.4])
    print("\n")
    NaiveBayesContinuo.print(NaiveBayesContinuo.normalizar(probabilidades))

    c = 0
    arch2 = pd.read_csv('iris.csv')
    for i in range(len(arch2)):
        filas2 = continuo.restituir(i,'iris.csv')
        prob2 = continuo.predecir(filas2)
        norm2 = NaiveBayesContinuo.normalizar(prob2)
        print(norm2)
        if list(norm2.values())[0] > list(norm2.values())[1] and list(norm2.values())[0] > list(norm2.values())[2]:
            d = "Iris-setosa"
        elif list(norm2.values())[1] > list(norm2.values())[0] and list(norm2.values())[1] > list(norm2.values())[2]:
            d = "Iris-versicolor"
        else:
            d = "Iris-virginica"
        if d == arch2.iloc[i,5]:
            c += 1

    print(c)
    pass


if __name__ == '__main__':
    main()
