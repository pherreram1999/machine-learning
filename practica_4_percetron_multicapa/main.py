import kagglehub as kg
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mlp import (entrenar, saveOnDisk, loadFromDisk, predecir)
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

import argparse

MODEL_NAME = 'iris.npz'

console = Console()

Y_map = {
    'Iris-setosa':    [1, 0, 0],
    'Iris-versicolor':[0, 1, 0],
    'Iris-virginica': [0, 0, 1],
}

ESPECIES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


def especie_desde_vector(v):
    idx = int(np.argmax(v))
    return ESPECIES[idx]


def load_data():
    kg.dataset_download("vikrishnan/iris-dataset", output_dir="data")
    df = pd.read_csv("data/iris.data.csv", header=None)
    Y = df[4].map(Y_map)
    X = df.iloc[:, :-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, Y, scaler


def entrenar_modelo():
    X, Y, scaler = load_data()
    W, B, ecm_hist = entrenar(X, Y, 4, 3, [128], 0.1, 5_000)
    saveOnDisk(MODEL_NAME, W, B, scaler.mean_, scaler.scale_)
    console.print(Panel(f"[green]Modelo guardado en {MODEL_NAME}[/green]\nECM final: {ecm_hist[-1]:.6f}"))


def ejeucion():
    x = np.zeros((1, 4))

    console.print(Panel("[bold cyan]Ingresa las características de la flor[/bold cyan]"))
    for i in range(4):
        x[0, i] = float(input(f'  X_{i}: '))

    W, B, scaler_mean, scaler_scale = loadFromDisk(MODEL_NAME)
    x_norm = (x - scaler_mean) / scaler_scale

    z = predecir(x_norm, W, B)
    z = z[0]

    z_escalonado = [1 if z[i] > 0.5 else 0 for i in range(3)]
    especie_pred = especie_desde_vector(z_escalonado)

    tabla = Table(title="Resultado de predicción", box=box.ROUNDED, show_lines=True)
    tabla.add_column("Especie",          style="cyan",   justify="left")
    tabla.add_column("Z real (MLP)",     style="yellow", justify="center")
    tabla.add_column("Z escalada (0/1)", style="green",  justify="center")

    for i in range(3):
        tabla.add_row(
            ESPECIES[i],
            f"{z[i]:.6f}",
            str(z_escalonado[i]),
        )

    console.print(tabla)
    console.print(Panel(f"[bold green]Predicción: {especie_pred}[/bold green]"))


def prueba_sustitucion():
    X, Y, _ = load_data()
    Y_list = list(Y)

    W, B, _, _ = loadFromDisk(MODEL_NAME)

    aciertos = 0
    tabla = Table(title="Prueba por sustitución", box=box.SIMPLE_HEAD, show_lines=False)
    tabla.add_column("#",               style="dim",    justify="right",  width=5)
    tabla.add_column("Esperado",        style="cyan",   justify="left")
    tabla.add_column("Predicho",        style="yellow", justify="left")
    tabla.add_column("Z raw (max)",     style="dim",    justify="center")
    tabla.add_column("Correcto",        style="bold",   justify="center")

    for p in range(len(X)):
        z = predecir(X[p].reshape(1, -1), W, B)[0]

        z_esc = [1 if z[i] > 0.5 else 0 for i in range(3)]
        esperado  = especie_desde_vector(Y_list[p])
        predicho  = especie_desde_vector(z_esc)
        correcto  = esperado == predicho
        if correcto:
            aciertos += 1

        marca = "[green]✓[/green]" if correcto else "[red]✗[/red]"
        tabla.add_row(
            str(p + 1),
            esperado,
            predicho,
            f"{float(np.max(z)):.4f}",
            marca,
        )

    porcentaje = 100 * aciertos / len(X)
    console.print(tabla)
    console.print(Panel(
        f"Aciertos: [bold]{aciertos}[/bold] / {len(X)}\n"
        f"Exactitud: [bold green]{porcentaje:.2f}%[/bold green]"
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--entrenar",  action="store_true", help="Entrenar modelo y guardar pesos/bias")
    parser.add_argument("-p", "--predecir",  action="store_true", help="Predecir especie ingresando valores")
    parser.add_argument("-t", "--test",      action="store_true", help="Prueba por sustitución con dataset completo")

    args = parser.parse_args()

    if args.entrenar:
        console.print("[bold]Entrenando modelo...[/bold]")
        return entrenar_modelo()
    if args.predecir:
        return ejeucion()
    if args.test:
        console.print("[bold]Prueba por sustitución...[/bold]")
        return prueba_sustitucion()


if __name__ == "__main__":
    main()
