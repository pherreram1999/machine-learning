from bayes import NaiveBayesDiscreto,NaiveBayesContinuo


def main():
    continuo = NaiveBayesContinuo()
    probabilidades = continuo.predecir([6.1,2.9,4.7,1.4])
    NaiveBayesContinuo.print(NaiveBayesContinuo.normalizar(probabilidades))

    pass


if __name__ == '__main__':
    main()
