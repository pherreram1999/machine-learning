import numpy as np
import os

def sigmoide(z):
    return 1/(1+np.exp(-z))

def d_sigmoide_z(z):
    s = sigmoide(z)
    return s*(1-s)

def entrenar(X, Yd, n_in,n_out, n_layers, lr, epoch_max):
    """
    Entrena el modelo segun los patrones dados
    :param X: matriz de caractericas de forma (muestras,caracteristicas)
    :param Yd: matriz de etiquetas, donde cada lista un clase , correspondiente al numero de muestras
    :param n_in: numero de neuronas entradas
    :param n_out: numero de neuronas de salida salidas
    :param n_layers: lista de numero de neuronas ocultas por cada capa, por ejemplop [2,5] seria 2 capas ocultas una de 2 neuronas, y la otra de 5
    :param lr: learning rate
    :param epoch_max: numero de epocas maximas
    :return: W (pesos),B (bias), ECM_historico
    """

    # contiene cuantas neuronas tiene cada capa
    # En un solo arreglo
    dimensiones = [n_in] + n_layers + [n_out]

    numero_dimensiones = len(dimensiones)

    numero_capas = len(dimensiones)

    # creamos vectores para almacenar vectores y pesos entre las capas
    # dentro contendran matrices de n x m entre capas para hacer la conexion

    W = [] # almacena los pesos entre las capas

    B = [] # almacena las bias entre las capas

    # incializamos los pesos, recordar que los pesos se conectan con las capas siguientes
    # es por ello que la union entre es una matriz de neuronas de la capa actual por
    # las neuronas de la capa siguiente

    # se mulitplica por valores chicos para que la valores generados sean chicos y la suma no se muy grande en un principio

    for i in range(numero_dimensiones - 1):
        W.append(
            np.random.randn(dimensiones[i], dimensiones[i + 1]) * 0.1
        )
        # El bias el valor que nos ayuda incializar los valores de la capa destino
        B.append(
            np.random.randn(dimensiones[i + 1]) * 0.1
        )


    # incializamos un valor alto
    ECM = 10 # nuestro error (distancia de Yobt respecto a Yd (dato real) )

    ECM_historico = [] # solo para registor grafico

    epoch = 0

    while ECM > 0.0 and epoch <= epoch_max:

        suma_ecm = 0

        for p in range(len(X)): # recorremos los patrones

            # ===== Propagacion hacia adelante

            # guardamos los valores de activacion, lo devuelto por nuestra funcion
            # de activacion, en caso del primer indice son valores de entrada
            A = [X[p]] #primer entrada como valores de activacion (Es la primera capa)
            # guardamos los valores antes de activar que se usaran para la derivada
            Z = []

            for i in range(len(W)):
                # recordar que es cada valor de activacion por cada peso
                z_actual = np.dot(A[i], W[i]) + B[i]
                Z.append(z_actual)
                A.append(sigmoide(z_actual))
                pass

            # la ultima activacion es la salida de esperada de Y obtenida
            # en este caso Y_obt es una lista de las etiquetas esperadas
            Y_obt = A[-1]

            # guardamos los errores para poder propagar hacia atras
            # los deltas son los errores temporales de cada capa
            deltas = [None] * len(W)

            # recordar que devuelve un arreglo de los errores (distancias)
            error = Yd[p] - Y_obt

            # Error cuadratico medio
            suma_ecm += np.sum(error**2)

            # la (Deseado - Obtenido) * derivada que conecta los errores entre capas
            # aplicamos la regla de la cadena para poder determinar el aporte de error de la cada neurona
            deltas[-1] = error * d_sigmoide_z(Z[-1]) # recordar que vamos hacia atras, por eso empezamos desde el final

            # propagamos el error desde ultimo capa hasta el principio
            # reverse itera un lista desde el final (la invierte) regresando un iterador
            for i in reversed(range(len(deltas) - 1)):
                # (Delta siguiente * Pesos) * derivada de Z actual
                deltas[i] =  np.dot(deltas[i+1], W[i+1].T) * d_sigmoide_z(Z[i])
                pass

            # actualizamos pesos
            for i in range(len(W)):
                W[i] += lr * np.outer(A[i], deltas[i])
                B[i] += lr * deltas[i]

            pass
        pass
        ECM = 0.5 * (suma_ecm / len(X))
        ECM_historico.append(ECM)
        epoch += 1

    pass


    return W,B, ECM_historico

def predecir(x, W, B):
    A = x
    for i in range(len(W)):
        # Z = (entrada * pesos) + bias
        z = np.dot(A, W[i]) + B[i]
        # La nueva activación es el resultado de la sigmoide
        A = sigmoide(z)
    return A


def saveOnDisk(name, W, B, scaler_mean=None, scaler_scale=None):
    """
    Guarda los datos entrenados en archivos en disco para su uso posterior
    """
    np.savez(
        name,
        W=np.array(W, dtype=object),
        B=np.array(B, dtype=object),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )



def loadFromDisk(name):
    """
    Carga los datos entrenados guardados en disco
    """
    if not os.path.exists(name):
            raise Exception(f"{name} no encontrado")

    with np.load(name, allow_pickle=True) as data:
        W = [np.array(w, dtype=float) for w in data['W']]
        B = [np.array(b, dtype=float) for b in data['B']]
        scaler_mean = data['scaler_mean'] if 'scaler_mean' in data.files else None
        scaler_scale = data['scaler_scale'] if 'scaler_scale' in data.files else None
        return W, B, scaler_mean, scaler_scale
