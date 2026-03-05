from bayes import NaiveBayes


def main():
    clasificador = NaiveBayes.discrete()
    clasificador.Ask(['S','F','A','F'])
    pass


if __name__ == '__main__':
    main()
