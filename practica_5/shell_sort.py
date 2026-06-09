"""
shell_sort.py
=============

Implementacion de Shellsort (ordenamiento por incrementos / "gaps") con un
contador de COSTO, mas las secuencias de gaps clasicas para comparar.

Shellsort en una frase:
  Es una mejora del ordenamiento por insercion. En lugar de comparar solo
  elementos vecinos (distancia 1), compara elementos separados por una
  distancia "gap". Con gaps grandes mueve elementos muy lejos de su lugar en
  pocos pasos; luego va reduciendo el gap hasta llegar a 1, donde hace una
  insercion clasica que ya solo tiene que arreglar desordenes pequeños.

Por que importan los gaps:
  La velocidad de Shellsort depende ENTERAMENTE de la secuencia de gaps usada.
  Mala secuencia -> casi tan lento como insercion (O(n^2)). Buena secuencia
  (Knuth, Ciura) -> mucho mas rapido. Por eso tiene sentido buscar gaps optimos
  con un algoritmo genetico.
"""

import numpy as np


def shell_sort(arr, gaps):
    """
    Ordena una copia de 'arr' usando Shellsort con la secuencia 'gaps' dada.

    Devuelve el arreglo ordenado y el COSTO. El costo NO es el tiempo de reloj
    (que es ruidoso y depende de la maquina), sino un conteo deterministico de
    operaciones: cada comparacion y cada movimiento suma 1. Ese numero es lo
    que el algoritmo genetico intenta minimizar.

    :param arr: arreglo de entrada (no se modifica; se trabaja sobre copia).
    :param gaps: lista/secuencia de gaps. DEBE terminar en 1 para garantizar
                 que el resultado quede totalmente ordenado.
    :return: (arreglo_ordenado: np.ndarray, costo: int)
    """
    # Copiamos para no mutar el arreglo original que nos pasaron.
    a = np.array(arr, copy=True)
    n = len(a)
    costo = 0

    # Recorremos cada gap de la secuencia (normalmente de mayor a menor).
    for gap in gaps:
        # Insercion por incrementos: para cada elemento desde la posicion 'gap',
        # lo "retrocedemos" de gap en gap hasta dejarlo en su lugar relativo.
        for i in range(gap, n):
            valor_actual = a[i]   # elemento que queremos colocar
            j = i

            # Mientras haya un elemento 'gap' posiciones atras que sea mayor,
            # lo desplazamos hacia adelante para hacer hueco.
            #   - La comparacion a[j - gap] > valor_actual cuenta como 1 costo.
            #   - Cada desplazamiento a[j] = a[j - gap] cuenta como 1 costo.
            while j >= gap and a[j - gap] > valor_actual:
                costo += 1            # costo de la comparacion que resulto True
                a[j] = a[j - gap]     # movimiento (desplazamiento)
                costo += 1            # costo del movimiento
                j -= gap

            # Si el while termino por la comparacion (no por j < gap), esa ultima
            # comparacion fallida tambien cuesta. La contamos solo cuando si se
            # llego a comparar (j >= gap).
            if j >= gap:
                costo += 1

            # Colocamos el valor en su posicion final dentro de este subarreglo.
            a[j] = valor_actual

    return a, costo
