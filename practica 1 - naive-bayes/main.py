import csv

from bayes import NaiveBayesDiscreto,NaiveBayesContinuo,NaiveBayes


def main():

    discreto = NaiveBayesDiscreto()

    for i in range(100):
        filas = discreto.restituir(i,'datos.csv')
        prob = discreto.predecir(filas)
        print(NaiveBayesDiscreto.normalizar(prob))

    continuo = NaiveBayesContinuo()
    probabilidades = continuo.predecir([6.1,2.9,4.7,1.4])
    NaiveBayesContinuo.print(NaiveBayesContinuo.normalizar(probabilidades))

    pass


if __name__ == '__main__':
    main()
