import kagglehub
import os
import pandas as pd
from Perceptron import Perceptron
from PerceptronGD import PerceptronGD
from PerceptronPSO import PerceptronPSO


def cargar_dataset():
    basepath = kagglehub.dataset_download(
        "uciml/breast-cancer-wisconsin-data",
        output_dir="data/brest-cancer"
    )

    csvpath = os.path.join(basepath, "data.csv")

    df = pd.read_csv(csvpath)

    # dataset de Kaggle trae columna "Unnamed: 32" toda NaN, hay que dropear
    # cualquier columna completamente vacia para que StandardScaler no genere NaN
    df = df.dropna(axis=1, how="all")
    df_original = df.copy()

    df.drop(columns=["id"], inplace=True)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    return X, y, df_original


def obtener_muestra(df_original, sample_id):
    fila = df_original[df_original["id"] == sample_id]
    if fila.empty:
        raise ValueError(f"No se encontró el ID {sample_id}")
    return fila.drop(columns=["id", "diagnosis"]).iloc[0]


def evaluar_rendimiento(perceptron, X, y, nombre=""):
    predicciones = perceptron.predecir(X)
    aciertos = (predicciones == y).sum()
    porcentaje = (aciertos / len(y)) * 100
    etiqueta = f" [{nombre}]" if nombre else ""
    print(f"Rendimiento por resustitución{etiqueta}: "
          f"{aciertos}/{len(y)} ({porcentaje:.2f}% de aciertos)")
    return porcentaje


def usar_clasico(X, y):
    # regla de delta (escalon, update por muestra)
    modelo = Perceptron(epochs=100, eta=0.01)
    modelo.entrenar(X, y)
    acc = evaluar_rendimiento(modelo, X, y, "Clasico")
    return modelo, acc


def usar_gd(X, y):
    # descenso gradiente heuristico (lineal, batch, targets {-1,+1})
    modelo = PerceptronGD(epochs=100, eta=0.01)
    modelo.entrenar(X, y)
    acc = evaluar_rendimiento(modelo, X, y, "GD")
    print(f"  Epocas usadas (early stopping): {len(modelo.costos)}")
    print(f"  Costo SSE inicial: {modelo.costos[0]:.4f}")
    print(f"  Costo SSE final:   {modelo.costos[-1]:.4f}")
    return modelo, acc


def usar_pso(X, y):
    # Particle Swarm Optimization, sin gradientes, fitness = errores
    modelo = PerceptronPSO(epochs=100, n_particulas=30,
                           w_inercia=0.7, c1=1.5, c2=1.5)
    modelo.entrenar(X, y)
    acc = evaluar_rendimiento(modelo, X, y, "PSO")
    print(f"  Iteraciones usadas: {len(modelo.historial_fitness)}")
    print(f"  Errores iniciales: {modelo.historial_fitness[0]}")
    print(f"  Errores finales:   {modelo.historial_fitness[-1]}")
    return modelo, acc


def predecir_muestra(modelo, df_original, nombre_modelo):
    print(f"\nUsando modelo: {nombre_modelo}")
    sample_id = int(input("Ingresa el ID del paciente para predecir: "))
    muestra = obtener_muestra(df_original, sample_id)

    # reshape a (1, n) porque StandardScaler requiere array 2D
    entrada = muestra.values.astype(float).reshape(1, -1)

    resultado = modelo.predecir(entrada)
    etiqueta = "Maligno (M)" if resultado[0] == 1 else "Benigno (B)"
    print(f"\nPrediccion: {etiqueta}")


def main():
    X, y, df_original = cargar_dataset()
    X_vals, y_vals = X.values, y.values

    print("\n--- Entrenando modelos ---\n")

    print("[1/3] Clasico")
    clasico, acc_clasico = usar_clasico(X_vals, y_vals)

    print("\n[2/3] Descenso gradiente")
    gd, acc_gd = usar_gd(X_vals, y_vals)

    print("\n[3/3] PSO")
    pso, acc_pso = usar_pso(X_vals, y_vals)

    print("\n--- Comparacion ---")
    print(f"Clasico: {acc_clasico:.2f}%")
    print(f"GD:      {acc_gd:.2f}%")
    print(f"PSO:     {acc_pso:.2f}%")

    # elegimos el mejor para la prediccion individual
    candidatos = [
        (acc_clasico, clasico, "Clasico"),
        (acc_gd, gd, "GD"),
        (acc_pso, pso, "PSO"),
    ]
    _, mejor, nombre_mejor = max(candidatos, key=lambda t: t[0])

    predecir_muestra(mejor, df_original, nombre_mejor)


if __name__ == "__main__":
    main()