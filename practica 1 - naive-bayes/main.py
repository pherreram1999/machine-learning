import csv

from bayes import NaiveBayesDiscreto,NaiveBayesContinuo,NaiveBayes


def main():

    discreto = NaiveBayesDiscreto()
    with open('datos.csv', mode='r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        for i, row in enumerate(lector):
            if(i > 0):
                prob = discreto.predecir([row[1],row[2],row[3],row[4]])
                print(NaiveBayesDiscreto.normalizar(prob))

    continuo = NaiveBayesContinuo()
    probabilidades = continuo.predecir([6.1,2.9,4.7,1.4])
    NaiveBayesContinuo.print(NaiveBayesContinuo.normalizar(probabilidades))

    pass


if __name__ == '__main__':
    main()
