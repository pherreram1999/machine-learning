import argparse  # módulo para parsear argumentos del CLI
from dataset import Dataset
from knn import KNN
from knn_ponderado import KNNPonderado


def ejecutar_clasico(valores_k: list[int]) -> None:
    """Ejecuta el KNN clásico (voto mayoritario) sobre el dataset Iris.csv."""
    dataset = Dataset("Iris.csv", "Species", ["Id"])
    dataset.info()
    print("\n" + "=" * 50)
    print("K-NN CLÁSICO - Iris (voto mayoritario)")
    print("=" * 50)
    for k in valores_k:
        modelo = KNN(k=k)
        modelo.entrenar(dataset.caracteristicas, dataset.clases)
        precision = modelo.evaluar_restitucion()
        print(f"  K={k:2d} -> Precisión por restitución: {precision:.2f}%")


def ejecutar_ponderado(valores_k: list[int]) -> None:
    """Ejecuta el KNN ponderado (Wi = (dk-di)/(dk-d1)) sobre el dataset Brest.csv."""
    dataset = Dataset("Brest.csv", "diagnosis", ["id"])
    dataset.info()
    print("\n" + "=" * 50)
    print("K-NN PONDERADO - Brest Cancer (Wi = (dk-di)/(dk-d1))")
    print("=" * 50)
    for k in valores_k:
        modelo = KNNPonderado(k=k)
        modelo.entrenar(dataset.caracteristicas, dataset.clases)
        precision = modelo.evaluar_restitucion()
        print(f"  K={k:2d} -> Precisión por restitución: {precision:.2f}%")


def main():
    """Función principal: parsea argumentos del CLI y ejecuta el modo elegido."""
    parser = argparse.ArgumentParser(description="Clasificador K-NN")

    # choice garantiza que se slecciona una de las 2 opciones
    parser.add_argument(
        "modo",
        choices=["clasico", "ponderado"],
        help="Variante de KNN a usar: 'clasico' (Iris) o 'ponderado' (Brest)",
    )
    parser.add_argument(
        "k_valores",
        help="Valores de k separados por comas (ej: 1,3,5,7)",
    )
    args = parser.parse_args()

    # convertir el string "1,3,5,7" a lista de enteros [1, 3, 5, 7]
    valores_k = [int(k) for k in args.k_valores.split(",")]

    if args.modo == "clasico":
        ejecutar_clasico(valores_k)
    else:
        ejecutar_ponderado(valores_k)


if __name__ == "__main__":
    main()