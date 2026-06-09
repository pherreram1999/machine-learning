import threading
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

MODEL_PATH = 'hand_landmarker.task'

COLORS = {
    ord('r'): (0,   0,   255),
    ord('g'): (0,   255,  0),
    ord('b'): (255,  0,   0),
    ord('y'): (0,   255, 255),
    ord('w'): (255, 255, 255),
    ord('k'): (0,    0,   0),
}

COLOR_NAMES = {
    ord('r'): 'rojo',
    ord('g'): 'verde',
    ord('b'): 'azul',
    ord('y'): 'amarillo',
    ord('w'): 'blanco',
    ord('k'): 'negro',
}


def only_index_up(landmarks):
    lm = landmarks
    index_up  = lm[8].y  < lm[6].y
    middle_dn = lm[12].y > lm[10].y
    ring_dn   = lm[16].y > lm[14].y
    pinky_dn  = lm[20].y > lm[18].y
    return index_up and middle_dn and ring_dn and pinky_dn


def blend_canvas(frame, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(bg, fg)


def draw_hud(frame, color_key, thickness, drawing):
    h, w = frame.shape[:2]
    color_bgr  = COLORS.get(color_key, (0, 0, 255))
    color_name = COLOR_NAMES.get(color_key, '?')

    cv2.rectangle(frame, (0, h - 38), (w, h), (30, 30, 30), -1)
    hud = "  r/g/b/y/w/k=color  +/-=grosor  c=limpiar  q=salir"
    cv2.putText(frame, hud, (4, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    state = f"Color: {color_name}  Grosor: {thickness}  {'[dibujando]' if drawing else ''}"
    cv2.putText(frame, state, (4, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1)
    cv2.rectangle(frame, (w - 28, h - 33), (w - 8, h - 13), color_bgr, -1)
    cv2.rectangle(frame, (w - 28, h - 33), (w - 8, h - 13), (200, 200, 200), 1)


class HandDrawApp:
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se pudo leer frame inicial")

        self.h, self.w = frame.shape[:2]
        self.canvas     = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.prev_point = None
        self.color_key  = ord('r')
        self.thickness  = 5
        self.drawing    = False

        # latest detection result (updated by callback)
        self._lock   = threading.Lock()
        self._result = None

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._ts = 0

    def _on_result(self, result, output_image, timestamp_ms):
        with self._lock:
            self._result = result

    def _get_result(self):
        with self._lock:
            return self._result

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._ts += 1
            self.detector.detect_async(mp_image, self._ts)

            result = self._get_result()
            self.drawing = False

            if result and result.hand_landmarks:
                lm = result.hand_landmarks[0]
                if only_index_up(lm):
                    x = int(lm[8].x * self.w)
                    y = int(lm[8].y * self.h)
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (x, y),
                                 COLORS[self.color_key], self.thickness)
                    self.prev_point = (x, y)
                    self.drawing = True
                else:
                    self.prev_point = None
            else:
                self.prev_point = None

            out = blend_canvas(frame, self.canvas)
            draw_hud(out, self.color_key, self.thickness, self.drawing)
            cv2.imshow('Nibbit Air Draw', out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas[:] = 0
            elif key in COLORS:
                self.color_key = key
            elif key in (ord('+'), ord('=')):
                self.thickness = min(self.thickness + 1, 30)
            elif key == ord('-'):
                self.thickness = max(self.thickness - 1, 1)

        self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    HandDrawApp().run()
