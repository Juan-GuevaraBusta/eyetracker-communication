"""
gaze.py
-------
Encapsula todo lo relacionado con la cámara y la extracción de la
"característica de mirada" (gaze feature) a partir de los landmarks
faciales de MediaPipe.

IMPORTANTE sobre la API utilizada:
La API antigua `mediapipe.solutions.face_mesh` fue ELIMINADA en las
versiones recientes de MediaPipe (>= 0.10.31). Este proyecto usa la
API vigente, "MediaPipe Tasks", a través de la clase
`mediapipe.tasks.python.vision.FaceLandmarker`, que entrega los mismos
478 puntos (incluye iris) pero requiere un archivo de modelo
(`face_landmarker.task`) que se descarga automáticamente la primera
vez que se ejecuta el programa.

Versión de MediaPipe utilizada: 0.10.21 (ver requirements.txt). Se probó
también la versión 1.0.1, pero en macOS/Apple Silicon presenta un bug
conocido que hace fallar (crash fatal) la subgráfica de detección de
rostro al inicializar el delegado de GPU, incluso pidiendo CPU
explícitamente; 0.10.21 no tiene ese problema y sigue usando la misma
API de Tasks (FaceLandmarker) vigente.

MEJORAS DE CALIBRACIÓN (basadas en documentación oficial de MediaPipe):

1. Contorno completo del ojo (16 puntos por ojo, en vez de solo 4
   esquinas). Los índices usados son los oficiales de
   `mediapipe.python.solutions.face_mesh_connections`
   (`FACEMESH_RIGHT_EYE` / `FACEMESH_LEFT_EYE`), documentados por
   Google. Usar el mínimo/máximo sobre todo el contorno para el ancho y
   alto del ojo es mucho más robusto ante el ruido de un landmark
   puntual que usar solo 2 esquinas.

2. Rechazo de parpadeos mediante blendshapes oficiales. Se activa
   `output_face_blendshapes=True` (documentado en
   `FaceLandmarkerOptions`, ver guía oficial "Face landmark detection"
   de Google AI Edge) y se usan las categorías estándar (compatibles
   con ARKit) `eyeBlinkLeft` / `eyeBlinkRight`. Durante un parpadeo la
   posición del iris no es representativa de la mirada real, así que
   esos frames se descartan tanto en calibración como en seguimiento.

3. Corrección de pose de cabeza (yaw/pitch) mediante la matriz de
   transformación facial. Se activa
   `output_facial_transformation_matrixes=True` (documentado en la
   misma guía oficial), que entrega una matriz 4x4 que "transforma los
   landmarks canónicos del rostro al rostro detectado". Se extrae la
   submatriz de rotación 3x3 y se calculan los ángulos yaw/pitch
   aproximados, que se agregan como 2 características adicionales.
   Así, la regresión de calibración puede aprender a compensar el
   efecto de pequeños giros de cabeza sobre la razón iris/ojo, en vez
   de depender únicamente de la normalización geométrica (sección 12
   del prompt: mitigar el movimiento de cabeza sin resolver head pose
   completo).
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.python.solutions import face_mesh_connections as _fmc

# ---------------------------------------------------------------------------
# Configuración de captura
# ---------------------------------------------------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ---------------------------------------------------------------------------
# Modelo de MediaPipe (se descarga una sola vez)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model_downloaded() -> str:
    """
    Verifica que el modelo face_landmarker.task exista localmente.
    Si no existe, lo descarga una única vez desde el repositorio oficial
    de modelos de MediaPipe (archivo estático, no un servicio/API externo
    que la aplicación consulte en tiempo de ejecución).

    Retorna la ruta local al archivo del modelo.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH

    print("Descargando modelo de MediaPipe (face_landmarker.task, ~3.7 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - queremos capturar cualquier fallo de red
        raise RuntimeError(
            "No se pudo descargar el modelo de MediaPipe. "
            "Verifica tu conexión a internet e intenta nuevamente.\n"
            f"Detalle: {exc}"
        ) from exc
    print("Modelo descargado correctamente en:", MODEL_PATH)
    return MODEL_PATH


# ---------------------------------------------------------------------------
# Índices de landmarks relevantes (malla facial de 478 puntos)
# ---------------------------------------------------------------------------
# Puntos de referencia (esquinas) usados para el sistema de coordenadas
# de cada ojo: esquina externa, esquina interna, párpado superior,
# párpado inferior.
EYE_A = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}   # un ojo
EYE_B = {"outer": 263, "inner": 362, "top": 386, "bottom": 374}  # el otro ojo

# Contorno COMPLETO de cada ojo (16 puntos), tomado de los conjuntos
# oficiales `FACEMESH_RIGHT_EYE` / `FACEMESH_LEFT_EYE` de
# `mediapipe.python.solutions.face_mesh_connections`. Se usa el
# min/max de todos estos puntos para estimar el ancho/alto del ojo de
# forma mucho más robusta que con solo 2 esquinas (un único landmark
# ruidoso afecta mucho menos al resultado).
EYE_A_CONTOUR = sorted(set(i for pair in _fmc.FACEMESH_RIGHT_EYE for i in pair))
EYE_B_CONTOUR = sorted(set(i for pair in _fmc.FACEMESH_LEFT_EYE for i in pair))

# Los landmarks de iris se agregan cuando refine_landmarks/output con iris
# está activo. Hay dos grupos de 5 puntos cada uno: [468-472] y [473-477].
# NO asumimos a priori cuál grupo corresponde a qué ojo (esa asignación
# puede variar entre versiones/documentación); en su lugar, cada grupo se
# asigna dinámicamente al ojo anatómico más cercano en el eje X (ver
# `_assign_iris_groups_to_eyes`).
IRIS_GROUP_1 = list(range(468, 473))
IRIS_GROUP_2 = list(range(473, 478))

NUM_LANDMARKS_WITH_IRIS = 478

# Categorías oficiales de blendshapes (compatibles con ARKit) usadas
# para rechazar frames con parpadeo. EYE_A se corresponde con las
# categorías "*Right*" y EYE_B con "*Left*", siguiendo la misma
# convención anatómica (subjetiva) que los índices de landmarks.
BLINK_CATEGORY_A = "eyeBlinkRight"
BLINK_CATEGORY_B = "eyeBlinkLeft"

# Umbral bajo para marcar casilla al parpadear de forma rápida.
# Sigue exigiendo AMBOS ojos > 0.25 (un solo ojo ruidoso no basta).
ENABLE_BLINK_REJECTION = True
BLINK_THRESHOLD = 0.25
BLINK_REQUIRE_BOTH_EYES = True

# Si False, yaw/pitch se mandan en 0 y no influyen en el mapper.
ENABLE_HEAD_POSE = True


def detect_blink(score_a, score_b, enabled=None, threshold=None, require_both=None):
    """True solo si el frame debe descartarse por parpadeo."""
    if enabled is None:
        enabled = ENABLE_BLINK_REJECTION
    if not enabled:
        return False
    if threshold is None:
        threshold = BLINK_THRESHOLD
    if require_both is None:
        require_both = BLINK_REQUIRE_BOTH_EYES
    if require_both:
        return score_a > threshold and score_b > threshold
    return score_a > threshold or score_b > threshold


def _landmark_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def _eye_center(landmarks, eye_def, w, h):
    outer = _landmark_xy(landmarks, eye_def["outer"], w, h)
    inner = _landmark_xy(landmarks, eye_def["inner"], w, h)
    return (outer + inner) / 2.0


def _iris_centroid(landmarks, group, w, h):
    pts = np.array([_landmark_xy(landmarks, i, w, h) for i in group])
    return pts.mean(axis=0)


def _assign_iris_groups_to_eyes(landmarks, w, h):
    """
    Determina, para el frame actual, cuál de los dos grupos de iris
    (IRIS_GROUP_1 / IRIS_GROUP_2) corresponde a EYE_A y cuál a EYE_B,
    comparando la coordenada X de sus centroides con la de cada ojo
    anatómico. Retorna (iris_center_A, iris_center_B).
    """
    center_a = _eye_center(landmarks, EYE_A, w, h)
    center_b = _eye_center(landmarks, EYE_B, w, h)

    iris_1 = _iris_centroid(landmarks, IRIS_GROUP_1, w, h)
    iris_2 = _iris_centroid(landmarks, IRIS_GROUP_2, w, h)

    # Distancia en X entre cada iris y cada ojo anatómico.
    d_1a = abs(iris_1[0] - center_a[0])
    d_1b = abs(iris_1[0] - center_b[0])

    if d_1a <= d_1b:
        iris_a, iris_b = iris_1, iris_2
    else:
        iris_a, iris_b = iris_2, iris_1

    return iris_a, iris_b


def _eye_feature(iris_center, contour_indices, landmarks, w, h):
    """
    Calcula la posición relativa (u, v) del iris dentro de un ojo,
    usando la caja delimitadora de TODO el contorno del ojo (16 puntos)
    en vez de solo 2 esquinas:

        u = (iris_x - min_x) / (max_x - min_x)
        v = (iris_y - min_y) / (max_y - min_y)

    Al ser razones dentro del propio ojo (y no coordenadas absolutas en
    la imagen), pequeños movimientos de cabeza tienden a cancelarse:
    si la cabeza se mueve, tanto el iris como el contorno del ojo se
    desplazan de forma similar en la imagen, y el cociente cambia poco.
    Usar el contorno completo (no solo 2 esquinas) reduce además el
    ruido de un landmark individual mal detectado.
    """
    pts = np.array([_landmark_xy(landmarks, i, w, h) for i in contour_indices])
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)

    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]

    # Evita división por cero si, por ruido de detección, los landmarks
    # del contorno casi coinciden.
    if abs(width) < 1e-6:
        width = 1e-6
    if abs(height) < 1e-6:
        height = 1e-6

    u = (iris_center[0] - min_xy[0]) / width
    v = (iris_center[1] - min_xy[1]) / height
    return u, v


def _blendshape_score(face_blendshapes, category_name):
    if not face_blendshapes:
        return 0.0
    for cat in face_blendshapes[0]:
        if cat.category_name == category_name:
            return float(cat.score)
    return 0.0


def _head_yaw_pitch(transform_matrix):
    """
    Extrae ángulos aproximados de yaw (giro izquierda/derecha) y pitch
    (giro arriba/abajo) a partir de la submatriz de rotación 3x3 de la
    matriz de transformación facial 4x4 que entrega MediaPipe (ver
    `output_facial_transformation_matrixes` en la guía oficial de
    Face Landmarker). Esta matriz "transforma los landmarks del modelo
    facial canónico al rostro detectado", por lo que su componente de
    rotación refleja la orientación de la cabeza respecto a la cámara.

    No se busca un ángulo físicamente exacto (para eso se necesitaría
    fijar una convención de ejes y validarla), sino una señal continua
    y consistente que la regresión de calibración pueda usar para
    compensar el efecto del giro de cabeza sobre la razón iris/ojo.
    """
    r = np.asarray(transform_matrix)[:3, :3]
    pitch = np.arcsin(np.clip(-r[1, 2], -1.0, 1.0))
    yaw = np.arctan2(r[0, 2], r[2, 2])
    return float(yaw), float(pitch)


class GazeTracker:
    """
    Envuelve la cámara web y el FaceLandmarker de MediaPipe. Cada llamada
    a `read()` captura un frame, ejecuta la detección y devuelve una
    característica de mirada de 6 valores:

        [u_der, v_der, u_izq, v_izq, yaw, pitch]

    además de un indicador de parpadeo (`blink`) para que el resto de
    la aplicación pueda descartar ese frame en vez de usar una posición
    de iris que no es representativa de la mirada real.
    """

    def __init__(self, camera_index=0):
        model_path = ensure_model_downloaded()

        # Se fuerza el delegado CPU explícitamente: en algunos entornos
        # (por ejemplo, sesiones sin acceso directo a la GPU/Metal) el
        # delegado por defecto puede fallar con un error fatal al crear
        # el servicio de GPU. CPU es más lento pero muchísimo más
        # portable, y para 640x480 en tiempo real es más que suficiente.
        base_options = BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU,
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,  # sección 18 del prompt: usar solo el rostro principal
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,             # para rechazo de parpadeos
            output_facial_transformation_matrixes=True,  # para compensar pose de cabeza
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                "No se pudo abrir la cámara web. Verifica que esté "
                "conectada, que no esté siendo usada por otra aplicación "
                "y que Python tenga permiso de acceso a la cámara."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self._frame_idx = 0

    def read(self):
        """
        Captura y procesa un frame.

        Retorna un diccionario con:
          - "ok": bool, si se pudo leer el frame de la cámara
          - "face_found": bool, si se detectó un rostro
          - "feature": np.ndarray[6] o None
          - "blink": bool, True si se detectó parpadeo en este frame
          - "blink_score_a" / "blink_score_b": scores de blendshape (0-1)
          - "frame_bgr": el frame capturado (BGR) o None
        """
        empty = {"ok": False, "face_found": False, "feature": None,
                 "blink": False, "blink_score_a": 0.0, "blink_score_b": 0.0,
                 "frame_bgr": None}
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            return empty

        frame_bgr = cv2.flip(frame_bgr, 1)  # efecto espejo, más intuitivo para el usuario
        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(self._frame_idx * (1000.0 / 30.0))
        self._frame_idx += 1

        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:  # noqa: BLE001 - robustez ante errores puntuales de MediaPipe
            return {"ok": True, "face_found": False, "feature": None,
                    "blink": False, "blink_score_a": 0.0, "blink_score_b": 0.0,
                    "frame_bgr": frame_bgr}

        if not result.face_landmarks:
            return {"ok": True, "face_found": False, "feature": None,
                    "blink": False, "blink_score_a": 0.0, "blink_score_b": 0.0,
                    "frame_bgr": frame_bgr}

        # Si por alguna razón se detecta más de un rostro, num_faces=1 ya
        # limita la salida al principal, pero nos protegemos igualmente.
        landmarks = result.face_landmarks[0]

        score_a = _blendshape_score(result.face_blendshapes, BLINK_CATEGORY_A)
        score_b = _blendshape_score(result.face_blendshapes, BLINK_CATEGORY_B)
        blink = detect_blink(score_a, score_b)

        if len(landmarks) < NUM_LANDMARKS_WITH_IRIS:
            # Landmarks incompletos (sin iris): no es posible calcular la
            # característica de mirada de forma confiable.
            return {"ok": True, "face_found": True, "feature": None,
                    "blink": blink, "blink_score_a": score_a, "blink_score_b": score_b,
                    "frame_bgr": frame_bgr}

        try:
            iris_a, iris_b = _assign_iris_groups_to_eyes(landmarks, w, h)
            u_a, v_a = _eye_feature(iris_a, EYE_A_CONTOUR, landmarks, w, h)
            u_b, v_b = _eye_feature(iris_b, EYE_B_CONTOUR, landmarks, w, h)

            # --- Corrección de pose de cabeza (yaw/pitch) -------------
            if ENABLE_HEAD_POSE and result.facial_transformation_matrixes:
                yaw, pitch = _head_yaw_pitch(result.facial_transformation_matrixes[0])
            else:
                yaw, pitch = 0.0, 0.0
        except Exception:  # noqa: BLE001
            return {"ok": True, "face_found": True, "feature": None,
                    "blink": blink, "blink_score_a": score_a, "blink_score_b": score_b,
                    "frame_bgr": frame_bgr}

        # [u_der, v_der, u_izq, v_izq, yaw, pitch] -> EYE_A se trata como
        # "derecho" y EYE_B como "izquierdo" de forma consistente en todo
        # el programa (el nombre es solo una etiqueta; lo importante es
        # la consistencia). yaw/pitch permiten que la regresión de
        # calibración compense pequeños giros de cabeza.
        feature = np.array([u_a, v_a, u_b, v_b, yaw, pitch], dtype=np.float64)

        return {"ok": True, "face_found": True, "feature": feature,
                "blink": blink, "blink_score_a": score_a, "blink_score_b": score_b,
                "frame_bgr": frame_bgr}

    def release(self):
        if self.cap is not None:
            self.cap.release()
        try:
            self._landmarker.close()
        except Exception:  # noqa: BLE001
            pass
