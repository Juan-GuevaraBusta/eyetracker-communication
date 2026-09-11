"""
calibration.py
--------------
Lógica de calibración y de la transformación "posición de los ojos ->
posición en pantalla".

¿Por qué es necesaria la calibración? (ver sección 5 del prompt)
Cada persona tiene una geometría facial distinta (tamaño de ojos,
distancia a la pantalla, posición de la cámara, etc.), por lo que no es
posible usar umbrales fijos como `if eye_x < 100`. En su lugar, se le
pide al usuario que mire 9 puntos conocidos de la pantalla; para cada
punto se registra la característica de mirada (ver gaze.py) y luego se
ajusta una función que aprende la relación entre esa característica y
la coordenada real en pantalla.

Modelo elegido: regresión lineal (afín) con regularización ridge muy
suave, resuelta con mínimos cuadrados. Es la opción más simple que
puede capturar la relación entre la posición relativa del iris (más
yaw/pitch de cabeza, ver gaze.py) y la posición en pantalla, tal como
sugiere la sección 9 del prompt.

Con la ampliación del vector de características a 6 valores
([u_der, v_der, u_izq, v_izq, yaw, pitch]) el modelo afín usa 7
parámetros por coordenada de pantalla (6 pesos + 1 sesgo). Con 9
puntos de calibración eso deja margen suficiente para que la
regularización ridge evite un ajuste inestable, sin sobre-parametrizar
el modelo (se evitó a propósito usar términos cuadráticos o de más
ojos, que con solo 9 puntos de calibración tienden a sobreajustar).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Puntos de calibración: 9 puntos en 20% / 50% / 80% del ancho y alto.
# ---------------------------------------------------------------------------
CALIBRATION_POINTS_PCT = [
    (0.20, 0.20), (0.50, 0.20), (0.80, 0.20),
    (0.20, 0.50), (0.50, 0.50), (0.80, 0.50),
    (0.20, 0.80), (0.50, 0.80), (0.80, 0.80),
]

# Puntos de validación: 5 puntos DISTINTOS a los de calibración
# (30% / 50% / 70%), para comprobar que el modelo generaliza y no solo
# "memorizó" los 9 puntos originales.
VALIDATION_POINTS_PCT = [
    (0.30, 0.30), (0.70, 0.30),
    (0.50, 0.50),
    (0.30, 0.70), (0.70, 0.70),
]

# Cuántas muestras se intentan capturar por punto (sección 7 del prompt:
# 20-50 muestras y promedio/mediana).
SAMPLES_PER_POINT = 40

# Tiempos del proceso de calibración, en segundos.
SECONDS_TO_LOOK_AT_POINT = 1.0   # tiempo para que el usuario dirija la mirada
SECONDS_CAPTURING = 1.5          # tiempo capturando muestras

# Como ahora se descartan los frames de parpadeo (blendshapes
# eyeBlinkLeft/eyeBlinkRight, ver gaze.py), es posible que en
# SECONDS_CAPTURING no se junten suficientes muestras válidas si el
# usuario parpadea varias veces. Se exige un mínimo de muestras
# válidas por punto y, si no se alcanza, se extiende la captura hasta
# este tiempo adicional máximo (evita quedarse esperando para siempre
# si el usuario deja de mirar la cámara).
MIN_VALID_SAMPLES_PER_POINT = 8
MAX_CAPTURE_EXTENSION_S = 4.0

# Mínimo de puntos agregados para poder ajustar el mapper (MVP: no
# abortar la calibración si se perdieron 1-2 puntos).
MIN_POINTS_TO_FIT = 3

# Umbral de error informado al usuario, expresado como % de la diagonal
# de la pantalla. En modo MVP no bloquea el paso a la grilla 3x3.
MAX_ERROR_PCT = 25.0

# Regularización ridge: evita que la matriz (X^T X) sea numéricamente
# inestable si algunas muestras de calibración son muy parecidas entre sí.
RIDGE_LAMBDA = 1e-3

# Factor de recorte por MAD (median absolute deviation) usado al agregar
# las muestras de un punto de calibración: una muestra se descarta si
# alguna de sus componentes está a más de este número de MADs de la
# mediana. 3.5 es un valor estándar y conservador para detección de
# outliers basada en MAD (equivalente aproximado a ~3 desviaciones
# estándar bajo una distribución normal).
MAD_OUTLIER_FACTOR = 3.5


def choose_point_samples(valid_samples, any_samples, min_valid=None):
    """
    Elige las muestras de un punto de calibración.

    Si hay suficientes muestras sin parpadeo, se usan esas. Si no, se
    usan todas las que tengan feature (incluye frames de parpadeo) para
    no descartar el punto completo.
    """
    if min_valid is None:
        min_valid = MIN_VALID_SAMPLES_PER_POINT
    if len(valid_samples) >= min_valid:
        return valid_samples
    if any_samples:
        return any_samples
    return valid_samples


def points_pct_to_pixels(points_pct, screen_w, screen_h):
    """Convierte una lista de puntos (pct_x, pct_y) a coordenadas de píxel."""
    return [(px * screen_w, py * screen_h) for px, py in points_pct]


class LinearGazeMapper:
    """
    Aprende y aplica la transformación:

        [u_der, v_der, u_izq, v_izq, yaw, pitch]  ->  (screen_x, screen_y)

    usando un modelo afín (lineal + término independiente) ajustado por
    mínimos cuadrados con una pequeña regularización ridge:

        theta = (X^T X + lambda*I)^-1 X^T Y

    El número de columnas de `features` es flexible (el código no
    asume 6 valores fijos), pero el proyecto siempre pasa el vector de
    6 valores producido por `gaze.py`.
    """

    def __init__(self):
        self.theta = None  # matriz (num_features + 1, 2): pesos + 1 sesgo, para x e y

    def fit(self, features, targets):
        """
        features: lista/array de forma (n, F) -> [u_der, v_der, u_izq, v_izq, yaw, pitch]
        targets:  lista/array de forma (n, 2) -> [screen_x, screen_y]
        """
        X = np.asarray(features, dtype=np.float64)
        Y = np.asarray(targets, dtype=np.float64)

        n = X.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        X_aug = np.hstack([X, ones])  # (n, 5)

        num_params = X_aug.shape[1]
        A = X_aug.T @ X_aug + RIDGE_LAMBDA * np.eye(num_params)
        b = X_aug.T @ Y
        self.theta = np.linalg.solve(A, b)  # (5, 2)

    def predict(self, feature):
        """feature: array-like de 4 valores. Retorna (screen_x, screen_y)."""
        if self.theta is None:
            raise RuntimeError("El modelo aún no ha sido calibrado (fit()).")
        x = np.asarray(feature, dtype=np.float64)
        x_aug = np.append(x, 1.0)  # (5,)
        pred = x_aug @ self.theta  # (2,)
        return float(pred[0]), float(pred[1])

    @property
    def is_fitted(self):
        return self.theta is not None


def compute_validation_error_pct(mapper, features, targets_px, screen_w, screen_h):
    """
    Calcula el error medio de predicción, expresado como porcentaje de
    la diagonal de la pantalla (sección 14 del prompt):

        error_pct = 100 * distancia_euclidiana_promedio / diagonal_pantalla

    Retorna (error_pct, lista_de_errores_individuales_px).
    """
    diagonal = float(np.hypot(screen_w, screen_h))
    errores = []
    for feat, (tx, ty) in zip(features, targets_px):
        px, py = mapper.predict(feat)
        dist = float(np.hypot(px - tx, py - ty))
        errores.append(dist)

    error_medio_px = float(np.mean(errores)) if errores else 0.0
    error_pct = 100.0 * error_medio_px / diagonal if diagonal > 0 else 0.0
    return error_pct, errores


def aggregate_samples(samples):
    """
    Combina varias muestras de la característica de mirada capturadas
    para un mismo punto en un único valor representativo.

    Se aplica un filtro de outliers robusto en dos pasos:

    1. Se calcula la mediana y la MAD (desviación absoluta respecto a
       la mediana) de cada componente del vector de características.
    2. Se descartan las muestras que se alejan más de
       `MAD_OUTLIER_FACTOR` MADs de la mediana en cualquier componente
       (por ejemplo, un parpadeo que se escapó del filtro de
       blendshapes en gaze.py, o un frame con detección ruidosa).
    3. Se recalcula la mediana solo con las muestras restantes.

    Este esquema (mediana + recorte por MAD) es más robusto ante
    valores atípicos puntuales que un simple promedio, y más robusto
    aún que una mediana simple si hay varios outliers agrupados hacia
    un lado (por ejemplo, varios frames seguidos de parpadeo).

    samples: lista de arrays de F valores (features de gaze.py).
    Retorna un array de F valores, o None si no hay muestras.
    """
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float64)  # (n, F)
    if arr.shape[0] < 4:
        # Muy pocas muestras para que el recorte por MAD sea confiable;
        # se usa directamente la mediana simple.
        return np.median(arr, axis=0)

    median = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - median), axis=0)
    # Evita división por cero cuando una componente es casi constante.
    mad_safe = np.where(mad < 1e-6, 1e-6, mad)

    deviations = np.abs(arr - median) / mad_safe
    keep_mask = np.all(deviations <= MAD_OUTLIER_FACTOR, axis=1)

    filtered = arr[keep_mask]
    if filtered.shape[0] < 3:
        # Si el filtro descartó casi todo (señal de que la mediana
        # inicial ya era poco representativa), se usa la mediana simple
        # sobre todos los datos originales en vez de quedarnos con muy
        # pocas muestras.
        return median

    return np.median(filtered, axis=0)
