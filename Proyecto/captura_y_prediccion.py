import cv2
import numpy as np
import os
import socket
import json
import argparse
from sklearn.neural_network import MLPClassifier
import joblib

def entrenar_o_cargar_modelo():
    """Carga el MLP si ya está entrenado y guardado, sino lo entrena desde el dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelo_path = os.path.join(base_dir, 'mlp_modelo.pkl')
    dataset_path = os.path.join(base_dir, 'dataset_geometrico_28x28.npz')
    
    if os.path.exists(modelo_path):
        print("Cargando modelo MLP guardado...")
        return joblib.load(modelo_path)
    
    print("Cargando dataset...")
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el archivo {dataset_path}.")
        return None
        
    data = np.load(dataset_path)
    X = data['X']
    y = data['Y']
    
    print("Entrenando Perceptron Multicapa (MLP)... Esto puede tardar un momento.")
    # Red simple para tareas como MNIST o este dataset de figuras
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, alpha=1e-4,
                        solver='adam', verbose=True, random_state=42)
    mlp.fit(X, y)
    print("Entrenamiento completado. Guardando modelo...")
    joblib.dump(mlp, modelo_path)
    return mlp

def procesar_trazo(lienzo):
    """
    Pipeline de preprocesamiento del trazo:
    1. Conversión a Escala de Grises (el lienzo ya es una matriz 2D de intensidades).
    2. Cálculo de Bounding Box.
    3. Recorte y Centrado en lienzo cuadrado perfecto.
    4. Redimensionamiento a 28x28.
    5. Binarización y Aplanamiento a vector 1D.
    """
    # 1. Escala de grises (El lienzo de entrada ya es escala de grises de 1 canal)
    
    # Buscar los píxeles activos para el Bounding Box
    coords = cv2.findNonZero(lienzo)
    if coords is None:
        return None
        
    # 2. Cálculo de la Caja Delimitadora (Bounding Box)
    x, y, w, h = cv2.boundingRect(coords)
    
    # 3. Recorte y Centrado
    recorte = lienzo[y:y+h, x:x+w]
    
    # Crear un nuevo lienzo cuadrado perfecto.
    # El tamaño será el lado máximo del Bounding Box más un margen para que no toque los bordes.
    lado_max = max(w, h)
    margen = max(int(lado_max * 0.2), 10) # 20% de margen
    lado_cuadrado = lado_max + margen * 2
    
    lienzo_cuadrado = np.zeros((lado_cuadrado, lado_cuadrado), dtype=np.uint8)
    
    # Calcular coordenadas para pegar el recorte centrado
    y_offset = (lado_cuadrado - h) // 2
    x_offset = (lado_cuadrado - w) // 2
    
    lienzo_cuadrado[y_offset:y_offset+h, x_offset:x_offset+w] = recorte
    
    # 4. Redimensionamiento a 28x28 píxeles
    resized = cv2.resize(lienzo_cuadrado, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 5. Binarización y Aplanamiento
    # Umbral (threshold) para forzar que cualquier píxel gris difuminado vuelva a ser 1 estricto
    _, binarizada = cv2.threshold(resized, 50, 1, cv2.THRESH_BINARY)
    
    # Aplanar la matriz 2D a un vector 1D (784 elementos)
    vector = binarizada.flatten()
    
    return vector, binarizada

def main(modo):
    mlp = entrenar_o_cargar_modelo()
    if mlp is None:
        return
        
    nombres_clases = ['Circulo', 'Triangulo', 'Cuadrado', 'Infinito', 'Letra V', 'Letra Z', 'Letra B', 'Letra N']

    # Configuracion del Socket UDP para enviar datos a Godot
    UDP_IP = "127.0.0.1"
    UDP_PORT = 12345
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Socket UDP configurado para enviar a Godot en {UDP_IP}:{UDP_PORT}")

    if modo == 'mouse':
        print("Iniciando modo MOUSE. Escuchando trazos de Godot en el puerto 12346...")
        sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_recv.bind((UDP_IP, 12346))
        
        # Enviar señal de listo a Godot
        try:
            ready_msg = json.dumps({"status": "ready"})
            sock.sendto(ready_msg.encode('utf-8'), (UDP_IP, UDP_PORT))
        except Exception as e:
            pass
            
        while True:
            try:
                data, addr = sock_recv.recvfrom(65535)
                mensaje = data.decode('utf-8')
                paquete = json.loads(mensaje)
                
                if "puntos" in paquete:
                    puntos = paquete["puntos"]
                    if len(puntos) < 2:
                        continue
                        
                    # Reconstruir lienzo
                    # Godot coordinates might be large, but let's just find the bounding box
                    # or draw directly on a 500x500 canvas.
                    # Actually, the points can just be drawn relative to their minimums.
                    puntos_np = np.array(puntos, dtype=np.int32)
                    min_x, min_y = np.min(puntos_np, axis=0)
                    max_x, max_y = np.max(puntos_np, axis=0)
                    
                    w = max_x - min_x + 20
                    h = max_y - min_y + 20
                    
                    # Normalizar puntos al nuevo lienzo (empezando en 10,10)
                    puntos_norm = puntos_np - [min_x - 10, min_y - 10]
                    
                    lienzo_dibujo = np.zeros((h, w), dtype=np.uint8)
                    cv2.polylines(lienzo_dibujo, [puntos_norm], isClosed=False, color=255, thickness=4)
                    
                    print("Trazo recibido de Godot. Procesando...")
                    resultado = procesar_trazo(lienzo_dibujo)
                    
                    if resultado is not None:
                        vector, img_bin = resultado
                        
                        vector_input = vector.reshape(1, -1)
                        probabilidades = mlp.predict_proba(vector_input)[0]
                        prediccion_idx = np.argmax(probabilidades)
                        confianza = probabilidades[prediccion_idx]
                        
                        UMBRAL_NINGUNA = 0.60
                        if confianza < UMBRAL_NINGUNA:
                            clase_detectada = "Ninguna"
                            id_clase = -1
                        else:
                            clase_detectada = nombres_clases[prediccion_idx]
                            id_clase = int(prediccion_idx)
                            
                        print(f"Prediccion (Mouse): {clase_detectada} ({confianza:.1%} conf)")
                        
                        paquete_udp = {
                            "figura": clase_detectada,
                            "id": id_clase,
                            "confianza": float(confianza)
                        }
                        
                        try:
                            mensaje_json = json.dumps(paquete_udp)
                            sock.sendto(mensaje_json.encode('utf-8'), (UDP_IP, UDP_PORT))
                        except Exception as e:
                            print(f"Error enviando paquete UDP: {e}")
                            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error en recepción UDP: {e}")
                
        sock_recv.close()
        return


    print("Iniciando captura de video...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    # Enviar señal de listo a Godot una vez encendida la cámara
    try:
        ready_msg = json.dumps({"status": "ready"})
        sock.sendto(ready_msg.encode('utf-8'), (UDP_IP, UDP_PORT))
    except Exception as e:
        pass

    lienzo_dibujo = None
    dibujando = False
    puntos_trazo = []
    frames_sin_detectar = 0
    LIMITE_FRAMES_APAGADO = 15 # Cantidad de frames consecutivos sin detectar el led para considerar que se apagó
    ultima_prediccion = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Efecto espejo
        if lienzo_dibujo is None:
            lienzo_dibujo = np.zeros(frame.shape[:2], dtype=np.uint8)
            
        # Convertir a HSV para detectar el rojo brillante del led/pluma
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Rango de color rojo en HSV mucho más estricto (alta saturación y alto brillo)
        # para evitar confundir con el color de la piel humana
        lower_red_1 = np.array([0, 200, 220])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 200, 220])
        upper_red_2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask_rojo = cv2.bitwise_or(mask1, mask2)
        
        # Suavizado para reducir ruido en la máscara
        mask_rojo = cv2.GaussianBlur(mask_rojo, (5, 5), 0)
        
        # Encontrar contornos activos en la máscara
        contornos, _ = cv2.findContours(mask_rojo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        punto_detectado = None
        
        if contornos:
            c_max = max(contornos, key=cv2.contourArea)
            # Solo tomarlo como led si tiene un area visible mínima
            if cv2.contourArea(c_max) > 10: 
                M = cv2.moments(c_max)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    punto_detectado = (cx, cy)
        
        # Lógica de dibujo en el lienzo
        if punto_detectado:
            frames_sin_detectar = 0
            # Dibujar un pequeño circulo verde en la pantalla principal para indicar dónde detectó el led
            cv2.circle(frame, punto_detectado, 5, (0, 255, 0), -1)
            
            if not dibujando:
                dibujando = True
                puntos_trazo = [punto_detectado]
            else:
                puntos_trazo.append(punto_detectado)
                # Trazar la línea en el lienzo de escala de grises (255 intensidad luminosa pura)
                cv2.line(lienzo_dibujo, puntos_trazo[-2], puntos_trazo[-1], 255, thickness=4)
                
        else:
            if dibujando:
                frames_sin_detectar += 1
                # Si pasa el limite de frames sin ver el rojo, consideramos que se apagó y procesamos
                if frames_sin_detectar > LIMITE_FRAMES_APAGADO:
                    print("LED apagado. Procesando el trazo...")
                    resultado = procesar_trazo(lienzo_dibujo)
                    
                    if resultado is not None:
                        vector, img_bin = resultado
                        
                        # Mostrar la imagen procesada de 28x28 (con zoom) en una ventana para depuración
                        cv2.imshow("Trazo Preprocesado (28x28)", cv2.resize(img_bin * 255, (200, 200), interpolation=cv2.INTER_NEAREST))
                        
                        # Predecir usando el Agente (MLP)
                        vector_input = vector.reshape(1, -1)
                        probabilidades = mlp.predict_proba(vector_input)[0]
                        prediccion_idx = np.argmax(probabilidades)
                        confianza = probabilidades[prediccion_idx]
                        
                        # Si la confianza es menor a un umbral razonable, decimos que no es ninguna
                        UMBRAL_NINGUNA = 0.60
                        if confianza < UMBRAL_NINGUNA:
                            ultima_prediccion = "Ninguna figura reconocida"
                            clase_detectada = "Ninguna"
                            id_clase = -1
                        else:
                            ultima_prediccion = f"{nombres_clases[prediccion_idx]} ({confianza:.1%} conf)"
                            clase_detectada = nombres_clases[prediccion_idx]
                            id_clase = int(prediccion_idx)
                            
                        print(f"Prediccion final: {ultima_prediccion}")
                        
                        # Construir el paquete y enviarlo por UDP a Godot
                        paquete_udp = {
                            "figura": clase_detectada,
                            "id": id_clase,
                            "confianza": float(confianza)
                        }
                        try:
                            mensaje_json = json.dumps(paquete_udp)
                            sock.sendto(mensaje_json.encode('utf-8'), (UDP_IP, UDP_PORT))
                            print(f"-> Paquete UDP enviado: {mensaje_json}")
                        except Exception as e:
                            print(f"Error enviando paquete UDP: {e}")
                    
                    # Reiniciar el lienzo para la siguiente figura
                    lienzo_dibujo = np.zeros(frame.shape[:2], dtype=np.uint8)
                    dibujando = False
                    puntos_trazo = []
        
        # Mezclar el lienzo de dibujo sobre el video de la camara para mostrar feedback al usuario
        mask_lienzo = lienzo_dibujo > 0
        frame[mask_lienzo] = [255, 0, 0] # Pintar el trazo de color azul en la visualización en vivo
        
        # Enviar frame a Godot por UDP
        # Redimensionar para reducir tamaño (Godot lo escalará al tamaño del TextureRect)
        frame_resized = cv2.resize(frame, (320, 240))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        result, encimg = cv2.imencode('.jpg', frame_resized, encode_param)
        if result:
            try:
                sock.sendto(encimg.tobytes(), (UDP_IP, 12347))
            except Exception as e:
                pass # Packet dropped or error
                
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detector de Figuras (Cámara o Mouse)")
    parser.add_argument('--modo', type=str, default='camara', choices=['camara', 'mouse'], help="Modo de entrada: camara o mouse")
    args = parser.parse_args()
    main(args.modo)
