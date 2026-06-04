import numpy as np
import cv2
import math
import random
import matplotlib.pyplot as plt

# Configuracion global
IMG_SIZE = 28
SAMPLES_PER_CLASS = 5000
# 0: Circulo, 1: Triangulo, 2: Cuadrado, 3: Infinito, 4: M, 5: Z, 6: B, 7: Alpha
NUM_CLASSES = 8


def draw_base_shape(shape_id):
    """Dibuja la figura base perfecta (solo contorno) en una matriz 28x28."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    center = (14, 14)

    if shape_id == 0:  # Circulo
        cv2.circle(img, center, 8, 1, thickness=1)

    elif shape_id == 1:  # Triangulo Equilatero
        # Calculo de vertices para centrarlo
        pts = np.array([[14, 6], [22, 20], [6, 20]], np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=1, thickness=1)

    elif shape_id == 2:  # Cuadrado
        cv2.rectangle(img, (6, 6), (22, 22), 1, thickness=1)

    elif shape_id == 3:  # Simbolo Infinito (Lemniscata)
        t = np.linspace(0, 2 * np.pi, 100)
        a = 8
        x = 14 + (a * math.sqrt(2) * np.cos(t)) / (np.sin(t) ** 2 + 1)
        y = 14 + (a * math.sqrt(2) * np.cos(t) * np.sin(t)) / (np.sin(t) ** 2 + 1)
        pts = np.column_stack((x, y)).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=1, thickness=1)

    elif shape_id == 4:  # Letra M
        pts = np.array([[6, 22], [6, 8], [14, 14], [22, 8], [22, 22]], np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=1, thickness=1)

    elif shape_id == 5:  # Letra Z
        pts = np.array([[6, 8], [22, 8], [6, 22], [22, 22]], np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=1, thickness=1)

    elif shape_id == 6:  # Letra B (Usando fuente de OpenCV)
        cv2.putText(img, 'B', (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1, thickness=1)


    elif shape_id == 7:  # Letra B (Usando fuente de OpenCV)
        cv2.putText(img, 'W', (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1, thickness=1)

    return img


def augment_image(img):
    """Aplica traslacion, rotacion, escalado y ruido aleatorio a la imagen."""
    # 1. Parametros aleatorios
    angle = random.uniform(-25, 25)  # Rotacion en grados (-25 a +25)
    scale = random.uniform(0.7, 1.2)  # Zoom out (0.7) y Zoom in (1.2)
    tx = random.randint(-4, 4)  # Traslacion X
    ty = random.randint(-4, 4)  # Traslacion Y

    # 2. Matriz de transformacion afin (Rotacion + Escala)
    center = (IMG_SIZE // 2, IMG_SIZE // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 3. Agregar traslacion a la matriz
    M[0, 2] += tx
    M[1, 2] += ty

    # 4. Aplicar transformacion
    # Usamos interpolacion lineal, pero luego debemos umbralizar para mantener ceros y unos
    augmented = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)

    # Asegurar que la matriz vuelva a ser estrictamente binaria (1 y 0) tras la interpolacion
    augmented = (augmented > 0.3).astype(np.uint8)

    # 5. Agregar Ruido Aleatorio (Salt & Pepper mode)
    # Volteamos aleatoriamente ~1.5% de los pixeles de la imagen
    noise_mask = np.random.rand(IMG_SIZE, IMG_SIZE) < 0.015
    augmented = np.logical_xor(augmented, noise_mask).astype(np.uint8)

    return augmented


def mostrar_figuras_originales():
    """Genera y plotea en pantalla las 8 figuras base sin alteraciones."""

    # Nombres descriptivos para los titulos de las graficas
    nombres = ['Círculo', 'Triángulo', 'Cuadrado', 'Infinito',
               'Letra M', 'Letra Z', 'Letra B', 'Alpha']

    # Creamos un lienzo (figure) con una cuadrícula de 2 filas y 4 columnas
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Figuras Base Originales (Sin Data Augmentation)", fontsize=16)

    for i in range(8):
        # 1. Generamos la matriz 28x28 de la figura original
        img_base = draw_base_shape(i)

        # 2. Calculamos sus coordenadas en la cuadrícula 2x4
        fila = i // 4
        columna = i % 4

        # 3. La dibujamos en su recuadro correspondiente
        axes[fila, columna].imshow(img_base, cmap='gray')
        axes[fila, columna].set_title(f"ID {i}: {nombres[i]}")
        axes[fila, columna].axis('off')  # Ocultamos los ejes X e Y para mayor limpieza

    # Ajustamos los márgenes para que no se superpongan los títulos
    plt.tight_layout()

    # Renderizamos la ventana emergente en pantalla
    plt.show()

def main():
    print("Iniciando la generacion del dataset (40,000 muestras en total)...")

    X_data = []
    y_data = []

    for class_id in range(NUM_CLASSES):
        base_img = draw_base_shape(class_id)

        for _ in range(SAMPLES_PER_CLASS):
            # Aplicar perturbaciones a la figura base
            img_aug = augment_image(base_img)

            # El paso crucial: Aplanar (Flattening) la matriz 28x28 a un vector 1D de 784 posiciones
            flattened_vector = img_aug.flatten()

            X_data.append(flattened_vector)
            y_data.append(class_id)

        print(f"Generadas {SAMPLES_PER_CLASS} muestras para la clase {class_id}.")

    # Convertir a matrices de NumPy. Usamos uint8 para ahorrar muchisima memoria RAM y espacio.
    X_data = np.array(X_data, dtype=np.uint8)
    y_data = np.array(y_data, dtype=np.uint8)

    # Guardar en disco duro de forma comprimida
    filename = 'dataset_geometrico_28x28.npz'
    np.savez_compressed(filename, X=X_data, Y=y_data)
    print(f"\n¡Dataset guardado con exito como '{filename}'!")
    print(f"Dimensiones de X (Entradas): {X_data.shape}")
    print(f"Dimensiones de Y (Etiquetas): {y_data.shape}")

    # --- CODIGO OPCIONAL PARA VISUALIZACION ---
    # Muestra 10 figuras al azar para que compruebes que parecen dibujadas a mano alzada
    fig, axes = plt.subplots(5, 5, figsize=(12, 5))
    fig.suptitle("Muestra aleatoria del Dataset Generado", fontsize=16)

    for ax in axes.flatten():
        idx = random.randint(0, len(X_data) - 1)
        # Reconstruimos el vector 1D a matriz 2D solo para poder visualizarlo
        img_2d = X_data[idx].reshape((IMG_SIZE, IMG_SIZE))
        ax.imshow(img_2d, cmap='gray')
        ax.set_title(f"Clase: {y_data[idx]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    mostrar_figuras_originales()