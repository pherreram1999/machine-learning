import argparse
from NaiveBayesDiscreto import NaiveBayesDiscreto
from NaiveBayesContinuo import NaiveBayesContinuo


def discreto():
    nv = NaiveBayesDiscreto()
    res = nv.predecir(['S', 'F', 'A', 'F'])
    print(res)
    print(nv.comprobar_por_restitucion())


def continuo():
    nv = NaiveBayesContinuo()
    res = nv.predecir([5.0, 3.4, 1.5, 0.2])
    print(res)
    print(nv.comprobar_por_restitucion())


def main():
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--discreto', action='store_true', help='Usar Naive Bayes Discreto')
    group.add_argument('-c', '--continuo', action='store_true', help='Usar Naive Bayes Continuo')
    args = parser.parse_args()

    if args.discreto:
        discreto()
    elif args.continuo:
        continuo()


if __name__ == "__main__":
    main()