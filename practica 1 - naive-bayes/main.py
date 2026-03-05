from bayes import NaiveBayesDiscreto


def main():
    discreto = NaiveBayesDiscreto()
    probabilidads = discreto.ask(['S','F','A','F'])


    print(probabilidads)
    pass


if __name__ == '__main__':
    main()
