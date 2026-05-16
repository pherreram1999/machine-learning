import kagglehub as kg
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mlp import (entrenar, saveOnDisk, loadFromDisk,predecir)

import argparse

MODEL_NAME = 'iris.npz'

def load_data():
    # cargamos el dataset
    kg.dataset_download("vikrishnan/iris-dataset", output_dir="data")
    df = pd.read_csv("data/iris.data.csv", header=None)

    # dicionario de valores para iris

    Y_map = {
        'Iris-setosa': [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1],
    }

    Y = df[4].map(Y_map)  # las etiquetas deben ser mapeedas a un arreglo
    # las caractersitcas las sacamos con un matriz de numpy
    X = df.iloc[:, :-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, Y, scaler


def entrenar_modelo():
    X, Y, scaler = load_data()
    W, B, _ = entrenar(X, Y, 4, 3, [128], 0.1, 5_000)
    saveOnDisk(MODEL_NAME, W, B, scaler.mean_, scaler.scale_)
    pass


def ejeucion():
    x = np.zeros((1,4))


    print('Ingreso los valores de las caracteristicas Xn')
    for i in range(4):
        x[0,i] = float(input(f'X_{i}: '))

    # cargamos el modelo del disco

    W, B, scaler_mean, scaler_scale = loadFromDisk(MODEL_NAME)

    # estandarizamos con media/std del set de entrenamiento
    x = (x - scaler_mean) / scaler_scale

    z = predecir(x, W, B)
    z = z[0] # scamos el primero valor, recordar que es una matriz, obtenemos el vector
    z_escalonado = [0] * 3
    # escalonamos valores para facil lectura

    for i in range(3):
        z_escalonado[i] = 1 if z[i] > 0.5 else 0


    print(z_escalonado,z)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--entrenar", action="store_true", help="entrenar modelo y guardar los pesos y bias")
    parser.add_argument("-p", "--predecir", action="store_true", help="epredecir con el modelo entrenado")

    args = parser.parse_args()

    if args.entrenar:
        print("==== Entrenando modelo ====")
        return entrenar_modelo()
    if args.predecir:
        print("==== Predecir con modelo ====")
        return ejeucion()


    pass


if __name__ == "__main__":
    main()