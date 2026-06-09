"""
benchmark.py
============

Script REPRODUCIBLE que genera la tabla comparativa del reporte: mide el costo
promedio de ordenar con Shellsort usando distintas secuencias de gaps clasicas
(Shell, Hibbard, Knuth, Sedgewick, Ciura) frente a los gaps que descubre el
algoritmo genetico.

Es autocontenido: las secuencias clasicas se definen aqui (se quitaron de
shell_sort.py para dejar el modulo enfocado solo en el genetico). Importa el
shell_sort instrumentado y la clase del GA desde la practica.

Uso:
    uv run python reporte/benchmark.py
"""

import os
import sys

import numpy as np

# Permitir importar los modulos de la practica (carpeta padre) al correr el
# script desde dentro de reporte/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datos
from shell_sort import shell_sort
from genetic_gaps import AlgoritmoGeneticoGaps


# ---------------------------------------------------------------------------
# Secuencias de gaps clasicas (de mayor a menor, terminando en 1)
# ---------------------------------------------------------------------------

def gaps_shell(n):
    """Shell (1959): n/2, n/4, ..., 1. La mas simple; peor caso O(n^2)."""
    gaps, gap = [], n // 2
    while gap > 0:
        gaps.append(gap)
        gap //= 2
    return gaps


def gaps_hibbard(n):
    """Hibbard (1963): 2^k - 1 -> 1, 3, 7, 15, ... Peor caso O(n^1.5)."""
    gaps, k = [], 1
    while (2 ** k - 1) < n:
        gaps.append(2 ** k - 1)
        k += 1
    return gaps[::-1] if gaps else [1]


def gaps_knuth(n):
    """Knuth: h = 3h + 1 -> 1, 4, 13, 40, ... ; equivale a (3^k - 1)/2."""
    gaps, h = [], 1
    while h < n:
        gaps.append(h)
        h = 3 * h + 1
    return gaps[::-1] if gaps else [1]


def gaps_sedgewick(n):
    """
    Sedgewick (1986): 1, 5, 19, 41, 109, ... Combina dos formulas:
      4^k + 3*2^(k-1) + 1, con 1 al inicio. Cota O(n^1.33) en peor caso.
    """
    gaps = [1]
    k = 1
    while True:
        g = 4 ** k + 3 * 2 ** (k - 1) + 1
        if g >= n:
            break
        gaps.append(g)
        k += 1
    return gaps[::-1]


def gaps_ciura(n):
    """Ciura (2001): 1, 4, 10, 23, 57, 132, 301, 701. Mejor empirica conocida."""
    base = [1, 4, 10, 23, 57, 132, 301, 701]
    gaps = [g for g in base if g < n]
    if 1 not in gaps:
        gaps.append(1)
    return gaps[::-1]


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------

def evaluar_gaps(gaps, arreglo_base):
    """
    Costo (comparaciones + movimientos) de ordenar el arreglo base.
    """
    _, costo = shell_sort(arreglo_base, gaps)
    return costo


def main():
    N = 2000
    arreglo_base = datos.cargar_arreglo(N)

    # El GA aprende sus gaps sobre el mismo arreglo base.
    ga = AlgoritmoGeneticoGaps(n=N, k_gaps=5, n_poblacion=40,
                               n_generaciones=60, n_muestras=1)
    gaps_ga = ga.entrenar()

    secuencias = {
        "Shell (n/2)":     gaps_shell(N),
        "Hibbard (2^k-1)": gaps_hibbard(N),
        "Knuth (3h+1)":    gaps_knuth(N),
        "Sedgewick":       gaps_sedgewick(N),
        "Ciura":           gaps_ciura(N),
        "GA (genetico)":   gaps_ga,
    }

    print(f"\nComparacion de secuencias de gaps  (N={N}, costo = comp.+movim., menor mejor)\n")
    print(f"{'Secuencia':<18}{'Costo prom.':>12}   Gaps")
    print("-" * 70)
    resultados = {}
    for nombre, gaps in secuencias.items():
        costo = evaluar_gaps(gaps, arreglo_base)
        resultados[nombre] = costo
        print(f"{nombre:<18}{costo:>12.1f}   {gaps}")

    mejor = min(resultados, key=resultados.get)
    print("-" * 70)
    print(f"Mejor: {mejor} ({resultados[mejor]:.1f})")


if __name__ == "__main__":
    main()
