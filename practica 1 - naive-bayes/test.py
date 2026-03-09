from bayes import NaiveBayesDiscreto


def discreto():
    nv = NaiveBayesDiscreto()
    res = nv.predecir(['S','F','A' ,'F'])
    print(res,res.max())

    pass

def main():
    discreto()
    pass

if __name__ == "__main__":
    main()