"""
datos.py
========

Utilidades para generar y persistir en disco el arreglo numerico desordenado
que usaremos como "caso base" del problema.

La idea de la practica es:
  1. Tener un arreglo de N elementos COMPLETAMENTE desordenado.
  2. Ordenarlo con Shellsort usando distintas secuencias de gaps.
  3. Dejar que un algoritmo genetico busque la secuencia de gaps que ordena
     ese arreglo con el menor costo posible.

Para que los resultados sean reproducibles (poder estudiarlos), el arreglo se
genera con una semilla fija y se guarda en disco (formato .npy de numpy), igual
que en practica_4 se guardaba el dataset iris en un .npz.
"""

import os
import numpy as np


# Ruta por defecto donde vive el arreglo desordenado.
RUTA_POR_DEFECTO = "data/arreglo_n.npy"


def crear_arreglo_desordenado(n: int,
                              ruta: str = RUTA_POR_DEFECTO,
                              random_seed: int = 1) -> np.ndarray:
    """
    Crea un arreglo de N enteros COMPLETAMENTE desordenado (en orden inverso) 
    y lo guarda en disco.
    """
    # np.arange(n - 1, -1, -1) genera [N-1, N-2, ..., 0] que es el orden opuesto.
    arreglo = np.arange(n - 1, -1, -1)

    # Aseguramos que exista la carpeta destino (p.ej. "data/") antes de guardar.
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)

    # np.save persiste el arreglo en formato binario .npy (rapido y exacto).
    np.save(ruta, arreglo)

    return arreglo


def cargar_arreglo(n: int,
                   ruta: str = RUTA_POR_DEFECTO,
                   random_seed: int = 1) -> np.ndarray:
    """
    Carga el arreglo desordenado desde disco; si no existe (o su tamaño no
    coincide con N), lo crea y lo guarda primero.

    Asi tanto el entrenamiento del genetico como la comparacion final usan
    EXACTAMENTE el mismo arreglo base.

    :param n: tamaño esperado del arreglo.
    :param ruta: archivo .npy de donde leer / donde guardar.
    :param random_seed: semilla usada si hay que crearlo.
    :return: el arreglo desordenado.
    """
    if os.path.exists(ruta):
        arreglo = np.load(ruta)
        # Si el archivo guardado tiene otro tamaño al pedido, lo regeneramos.
        if arreglo.shape[0] == n:
            return arreglo

    return crear_arreglo_desordenado(n, ruta, random_seed)
