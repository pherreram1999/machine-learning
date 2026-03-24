import argparse  # módulo para parsear argumentos del CLI
from dataset import Dataset
from knn import KNN
from knn_ponderado import KNNPonderado


def ejecutar_clasico(ruta_csv: str, valores_k: list[int]) -> None:
    """Ejecuta el KNN clásico (voto mayoritario) con los valores de k dados."""
    dataset = Dataset(ruta_csv, "Species", ["Id"])
    dataset.info()
    print("\n" + "=" * 50)
    print("K-NN CLÁSICO (voto mayoritario)")
    print("=" * 50)
    for k in valores_k:
        modelo = KNN(k=k)
        modelo.entrenar(dataset.caracteristicas, dataset.clases)
        precision = modelo.evaluar_restitucion()
        print(f"  K={k:2d} -> Precisión por restitución: {precision:.2f}%")


def ejecutar_ponderado(ruta_csv: str, valores_k: list[int]) -> None:
    """Ejecuta el KNN ponderado (peso = 1/distancia) con los valores de k dados."""
    dataset = Dataset(ruta_csv, "Species", ["Id"])
    dataset.info()
    print("\n" + "=" * 50)
    print("K-NN PONDERADO (peso = 1/distancia)")
    print("=" * 50)
    for k in valores_k:
        modelo = KNNPonderado(k=k)
        modelo.entrenar(dataset.caracteristicas, dataset.clases)
        precision = modelo.evaluar_restitucion()
        print(f"  K={k:2d} -> Precisión por restitución: {precision:.2f}%")


def main():
    """Función principal: parsea argumentos del CLI y ejecuta el modo elegido."""
    parser = argparse.ArgumentParser(description="Clasificador K-NN")
    parser.add_argument(
        "modo",
        choices=["clasico", "ponderado"],
        help="Variante de KNN a usar: 'clasico' o 'ponderado'",
    )
    parser.add_argument(
        "dataset",
        help="Ruta o nombre del archivo CSV del dataset",
    )
    parser.add_argument(
        "k_valores",
        help="Valores de k separados por comas (ej: 1,3,5,7)",
    )
    args = parser.parse_args()

    # convertir el string "1,3,5,7" a lista de enteros [1, 3, 5, 7]
    valores_k = [int(k) for k in args.k_valores.split(",")]

    if args.modo == "clasico":
        ejecutar_clasico(args.dataset, valores_k)
    else:
        ejecutar_ponderado(args.dataset, valores_k)


if __name__ == "__main__":
    main()