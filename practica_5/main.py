"""
main.py
=======

Orquestacion de la practica 5:
  - Carga (o crea) un arreglo desordenado de tamaño N guardado en disco.
  - Entrena un algoritmo genetico para hallar gaps optimos de Shellsort.
  - Compara los gaps del genetico contra secuencias clasicas (Shell, Knuth, Ciura).
  - Demuestra que el arreglo queda realmente ordenado.
"""

import numpy as np

import datos
from shell_sort import shell_sort
from genetic_gaps import AlgoritmoGeneticoGaps


def main():
    # --- Parametros del experimento ---
    N = 3000         # tamaño del arreglo desordenado (opuesto)
    K_GAPS = 5       # cuantos gaps maneja cada cromosoma del genetico

    # Arreglo base desordenado (se crea y guarda en disco si no existe).
    arreglo_base = datos.cargar_arreglo(N)
    print(f"Arreglo desordenado de N={N} cargado desde disco (data/arreglo_n.npy)")

    # --- Entrenamiento del algoritmo genetico ---
    print("\n--- Entrenando algoritmo genetico para hallar gaps optimos ---")
    ga = AlgoritmoGeneticoGaps(n=N, k_gaps=K_GAPS,
                               n_poblacion=40, n_generaciones=60,
                               n_muestras=1)
    gaps_ga = ga.entrenar()

    print(f"Gaps encontrados por el GA: {gaps_ga}")
    print(f"  Costo promedio inicial (gen 0):   {ga.historial_fitness[0]:.1f}")
    print(f"  Costo promedio final (mejor):     {ga.mejor_costo:.1f}")

    # --- Demostracion: ordenar el arreglo de disco con los gaps del GA ---
    print("\n--- Demostracion de ordenamiento con los gaps del GA ---")
    ordenado, costo = shell_sort(arreglo_base, gaps_ga)
    print(f"Primeros 10 antes:   {arreglo_base[:10]}")
    print(f"Primeros 10 despues: {ordenado[:10]}")
    correcto = np.array_equal(ordenado, np.sort(arreglo_base))
    print(f"Ordenado correctamente? {correcto}   (costo de esta corrida: {costo})")


if __name__ == "__main__":
    main()
