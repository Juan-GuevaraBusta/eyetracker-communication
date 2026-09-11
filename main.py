"""
main.py
-------
Punto de entrada del eye tracker académico. Controla:

  * la ventana en PANTALLA COMPLETA REAL,
  * la máquina de estados (instrucciones -> calibración -> validación ->
    seguimiento),
  * el dibujo de la cuadrícula 3x3 y del cuadro resaltado,
  * la estabilización de la predicción,
  * el HUD (gaze, cuadro, FPS, estado),
  * los controles de teclado (ESC / R / SPACE).

Ejecutar con:  python main.py
"""

import sys
import time
from collections import deque, Counter

import cv2
import numpy as np

import calibration as calib
from gaze import GazeTracker, detect_blink

# ---------------------------------------------------------------------------
# Constantes de la interfaz
# ---------------------------------------------------------------------------
WINDOW_NAME = "Eye Tracker 3x3"

GRID_LINE_THICKNESS = 6
GRID_LINE_COLOR = (0, 0, 0)          # negro, sección 2 y 4 del prompt
SELECTED_FILL_COLOR = (220, 90, 0)   # azul BGR: cuadro elegido al parpadear

# Frases de la grilla, fila por fila (celdas 1-9).
CELL_LABELS = {
    1: "tengo hambre",
    2: "dejenme solo",
    3: "tengo sed",
    4: "si",
    5: "como estas",
    6: "no",
    7: "arriba",
    8: "me duele",
    9: "abajo",
}
CELL_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

STABILITY_WINDOW = 10                # últimas N predicciones (sección 11)
SMOOTHING_ALPHA = 0.35               # suavizado exponencial del punto de mirada

PREVIEW_W, PREVIEW_H = 220, 165      # miniatura de la cámara en pantalla

# Marco negro alrededor de la grilla al parpadear (no cambia el gaze).
BLINK_FRAME_HOLD_S = 0.35
BLINK_FRAME_MARGIN = 56

# Estados de la máquina de estados
STATE_INSTRUCTIONS = "instructions"
STATE_CALIBRATING = "calibrating"
STATE_VALIDATING = "validating"
STATE_TRACKING = "tracking"
STATE_QUIT = "quit"


# ---------------------------------------------------------------------------
# Utilidades de pantalla completa
# ---------------------------------------------------------------------------
def get_screen_resolution():
    """
    Obtiene la resolución REAL del monitor principal para que el lienzo
    de la interfaz se dibuje exactamente a ese tamaño (y no a un valor
    fijo como 1280x720), tal como exige la sección 4 del prompt.
    """
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        if monitors:
            m = monitors[0]
            if m.width and m.height:
                return int(m.width), int(m.height)
    except Exception:
        pass

    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w and h:
            return int(w), int(h)
    except Exception:
        pass

    # Última alternativa si no se pudo detectar el monitor.
    return 1280, 720


# ---------------------------------------------------------------------------
# Dibujo de la cuadrícula
# ---------------------------------------------------------------------------
def build_grid_canvas(w, h):
    """Prerenderiza la cuadrícula 3x3 UNA sola vez (optimización de FPS)."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for i in (1, 2):
        x = int(w * i / 3)
        cv2.line(canvas, (x, 0), (x, h), GRID_LINE_COLOR, GRID_LINE_THICKNESS)
        y = int(h * i / 3)
        cv2.line(canvas, (0, y), (w, y), GRID_LINE_COLOR, GRID_LINE_THICKNESS)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), GRID_LINE_COLOR, GRID_LINE_THICKNESS)
    return canvas


def cell_bounds(cell_idx, w, h):
    """cell_idx en 1..9 -> (x0, y0, x1, y1) en píxeles."""
    row = (cell_idx - 1) // 3
    col = (cell_idx - 1) % 3
    x0 = int(col * w / 3)
    x1 = int((col + 1) * w / 3)
    y0 = int(row * h / 3)
    y1 = int((row + 1) * h / 3)
    return x0, y0, x1, y1


def compute_cell(x, y, w, h):
    """
    Divide la pantalla en 3 columnas y 3 filas (sección 10 del prompt) y
    devuelve el número de cuadro (1-9) correspondiente a (x, y).
    """
    col = int(x / (w / 3.0))
    row = int(y / (h / 3.0))
    col = max(0, min(2, col))
    row = max(0, min(2, row))
    return row * 3 + col + 1


def next_selected_cell(selected, gazed, blink_now, blink_prev):
    """El relleno azul solo cambia en el flanco de subida del parpadeo."""
    if gazed is None:
        return selected
    if blink_now and not blink_prev:
        return gazed
    return selected


def gaze_cell_for_blink_mark(last_open, instant, eyes_closed):
    """
    Al parpadear el iris se deforma y la predicción de ESE frame miente.
    Se marca la última casilla con ojos abiertos (donde iba el punto).
    """
    if eyes_closed:
        return last_open
    return instant


def draw_selected_cell(canvas, cell_idx, w, h):
    """Rellena de azul el cuadro elegido (uno a la vez)."""
    if cell_idx is None:
        return
    x0, y0, x1, y1 = cell_bounds(cell_idx, w, h)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), SELECTED_FILL_COLOR, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), GRID_LINE_COLOR, GRID_LINE_THICKNESS)


def label_layout_for_cell(text, cell_w, cell_h):
    """
    Parte la frase en líneas (una palabra por línea si hay varias) y
    elige la escala más grande que cabe en ~86% del ancho y ~62% del alto.
    """
    lines = text.split() if " " in text else [text]
    max_w = cell_w * 0.86
    max_h = cell_h * 0.62
    scale = 4.0
    thickness = 6
    while scale >= 0.4:
        thickness = max(2, int(round(scale * 2.6)))
        widths, heights = [], []
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, CELL_LABEL_FONT, scale, thickness)
            widths.append(tw)
            heights.append(th)
        gap = int(max(8, scale * 16))
        total_h = sum(heights) + gap * (len(lines) - 1)
        if max(widths) <= max_w and total_h <= max_h:
            return lines, float(scale), int(thickness)
        scale -= 0.06
    thickness = max(2, int(round(scale * 2.6)))
    return lines, float(max(scale, 0.4)), int(thickness)


def draw_cell_labels(canvas, w, h, selected_cell=None):
    """Dibuja las frases centradas en cada casilla, encima del relleno azul."""
    for idx, text in CELL_LABELS.items():
        x0, y0, x1, y1 = cell_bounds(idx, w, h)
        cw, ch = x1 - x0, y1 - y0
        lines, scale, thickness = label_layout_for_cell(text, cw, ch)
        sizes = [cv2.getTextSize(line, CELL_LABEL_FONT, scale, thickness)[0] for line in lines]
        gap = int(max(8, scale * 16))
        total_h = sum(s[1] for s in sizes) + gap * (len(lines) - 1)
        color = (255, 255, 255) if idx == selected_cell else (20, 20, 20)
        outline = (0, 0, 0) if idx == selected_cell else (255, 255, 255)
        y = y0 + (ch - total_h) // 2 + sizes[0][1]
        cx = (x0 + x1) // 2
        for line, (tw, th) in zip(lines, sizes):
            x = cx - tw // 2
            outline_th = thickness + 3
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                cv2.putText(canvas, line, (x + dx, y + dy), CELL_LABEL_FONT,
                            scale, outline, outline_th, cv2.LINE_AA)
            cv2.putText(canvas, line, (x, y), CELL_LABEL_FONT,
                        scale, color, thickness, cv2.LINE_AA)
            y += th + gap


def blink_frame_active(now, last_blink_at, hold_s=BLINK_FRAME_HOLD_S):
    """True si debe mostrarse el marco negro (parpadeo o hold breve)."""
    if last_blink_at is None:
        return False
    return (now - last_blink_at) <= hold_s


def draw_blink_frame(canvas, margin=BLINK_FRAME_MARGIN):
    """Pinta un marco negro alrededor de la grilla; el interior no se toca."""
    h, w = canvas.shape[:2]
    m = int(margin)
    if m <= 0:
        return
    black = (0, 0, 0)
    cv2.rectangle(canvas, (0, 0), (w - 1, m), black, -1)
    cv2.rectangle(canvas, (0, h - m), (w - 1, h - 1), black, -1)
    cv2.rectangle(canvas, (0, 0), (m, h - 1), black, -1)
    cv2.rectangle(canvas, (w - m, 0), (w - 1, h - 1), black, -1)


def draw_text_block(canvas, lines, org=(30, 40), scale=0.8, color=(0, 0, 0),
                     thickness=2, line_gap=34, bg=True):
    x, y = org
    if bg and lines:
        widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for t in lines]
        box_w = max(widths) + 24
        box_h = line_gap * len(lines) + 16
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x - 12, y - 30), (x - 12 + box_w, y - 30 + box_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, dst=canvas)
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_gap


def draw_camera_preview(canvas, frame_bgr, face_found, w):
    """Miniatura de la cámara en la esquina superior derecha (feedback visual)."""
    if frame_bgr is None:
        return
    thumb = cv2.resize(frame_bgr, (PREVIEW_W, PREVIEW_H))
    x0 = w - PREVIEW_W - 20
    y0 = 20
    canvas[y0:y0 + PREVIEW_H, x0:x0 + PREVIEW_W] = thumb
    border_color = (60, 180, 60) if face_found else (0, 0, 220)
    cv2.rectangle(canvas, (x0, y0), (x0 + PREVIEW_W, y0 + PREVIEW_H), border_color, 3)
    label = "Rostro OK" if face_found else "Sin rostro"
    cv2.putText(canvas, label, (x0, y0 + PREVIEW_H + 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, border_color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Estabilización (sección 11 del prompt)
# ---------------------------------------------------------------------------
class Stabilizer:
    """
    Guarda las últimas N predicciones de cuadro y devuelve la que aparece
    con más frecuencia (voto mayoritario). Esto evita que pequeñas
    variaciones frame a frame hagan "parpadear" la interfaz entre dos
    cuadros vecinos.
    """

    def __init__(self, window=STABILITY_WINDOW):
        self.history = deque(maxlen=window)

    def update(self, cell):
        self.history.append(cell)
        return Counter(self.history).most_common(1)[0][0]

    def reset(self):
        self.history.clear()


class FpsMeter:
    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last_t = time.time()

    def tick(self):
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        return self.fps


# ---------------------------------------------------------------------------
# Aplicación principal
# ---------------------------------------------------------------------------
class EyeTrackerApp:
    def __init__(self):
        self.screen_w, self.screen_h = get_screen_resolution()
        self.grid_canvas = build_grid_canvas(self.screen_w, self.screen_h)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.tracker = None
        self.camera_error = None
        try:
            self.tracker = GazeTracker()
        except Exception as exc:  # cámara no disponible (sección 18)
            self.camera_error = str(exc)

        self.mapper = calib.LinearGazeMapper()
        self.stabilizer = Stabilizer()
        self.smoothed_point = None
        self.fps_meter = FpsMeter()
        self.last_error_pct = None

    # -- helpers de lectura ------------------------------------------------
    def _read(self):
        """Lee un frame de la cámara; None si la cámara se desconectó."""
        result = self.tracker.read()
        if not result["ok"]:
            return None
        return result

    def _blank_canvas(self):
        return np.full((self.screen_h, self.screen_w, 3), 255, dtype=np.uint8)

    def _window_closed(self):
        """Detecta si la ventana fue cerrada por el usuario (ej. Cmd+W / la X
        del sistema), para poder salir limpiamente (sección 18 del prompt)."""
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    # -- estado: instrucciones ---------------------------------------------
    def _state_instructions(self):
        instrucciones = [
            "CALIBRACION",
            "",
            "1. Sientate frente a la camara.",
            "2. Manten la cabeza relativamente quieta.",
            "3. Manten una distancia constante de la pantalla.",
            "4. Asegurate de que tu rostro este bien iluminado.",
            "5. No muevas la camara despues de calibrar.",
            "6. Sigue los puntos unicamente con los ojos.",
            "7. Mira directamente cada punto cuando aparezca.",
            "",
            "Presiona ESPACIO para comenzar.",
            "ESC para salir.",
        ]
        while True:
            data = self._read()
            if data is None:
                return STATE_QUIT, "Se perdió la conexión con la cámara."

            canvas = self._blank_canvas()
            draw_text_block(canvas, instrucciones, org=(40, 60), scale=0.85, line_gap=38)

            if not data["face_found"]:
                draw_text_block(
                    canvas,
                    ["No se detecta el rostro.", "Ajusta tu posicion frente a la camara."],
                    org=(40, self.screen_h - 120), scale=0.9, color=(0, 0, 200), line_gap=40,
                )

            draw_camera_preview(canvas, data["frame_bgr"], data["face_found"], self.screen_w)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or self._window_closed():  # ESC o cierre de ventana
                return STATE_QUIT, None
            if key == 32 and data["face_found"]:  # SPACE, solo si hay rostro
                return STATE_CALIBRATING, None

    # -- captura de un punto (compartida por calibración y validación) -----
    def _capture_point(self, point_px, look_seconds, capture_seconds, label, canvas_base):
        """
        Muestra un punto en `point_px`, espera `look_seconds` para que el
        usuario dirija la mirada y luego captura muestras durante
        `capture_seconds`. Retorna las features del punto (sin parpadeo
        si hay suficientes; si no, fallback a cualquier frame con
        feature para no perder el punto).
        """
        px, py = int(point_px[0]), int(point_px[1])
        t_start = time.time()
        phase = "look"
        valid_samples = []
        any_samples = []

        while True:
            data = self._read()
            if data is None:
                return None  # cámara perdida

            elapsed = time.time() - t_start
            if phase == "look" and elapsed >= look_seconds:
                phase = "capture"
                t_start = time.time()
                elapsed = 0.0

            canvas = canvas_base.copy()
            draw_text_block(canvas, [label], org=(40, 60), scale=0.9, line_gap=36)

            # Círculo del punto: se encoge durante "look" y queda fijo
            # (parpadeando levemente) durante "capture".
            if phase == "look":
                shrink = max(0.0, 1.0 - elapsed / max(look_seconds, 1e-6))
                radius = int(14 + 26 * shrink)
                color = (0, 0, 220)
            else:
                radius = 16 if int(elapsed * 4) % 2 == 0 else 12
                color = (0, 140, 255)

            cv2.circle(canvas, (px, py), radius, color, -1)
            cv2.circle(canvas, (px, py), radius + 4, (0, 0, 0), 2)

            capturing = (
                phase == "capture"
                and elapsed <= capture_seconds + calib.MAX_CAPTURE_EXTENSION_S
            )
            if capturing and data["face_found"] and data["feature"] is not None:
                any_samples.append(data["feature"])
                if not data["blink"]:
                    valid_samples.append(data["feature"])

            extending = phase == "capture" and elapsed > capture_seconds
            debug_lines = [
                f"Muestras validas: {len(valid_samples)}/{calib.MIN_VALID_SAMPLES_PER_POINT}",
                f"Muestras totales: {len(any_samples)}",
                f"Blink R: {data.get('blink_score_a', 0.0):.2f}  L: {data.get('blink_score_b', 0.0):.2f}",
                f"Parpadeo: {'SI' if data.get('blink') else 'no'}",
                f"Fase: {phase}{' +extra' if extending else ''}",
            ]
            draw_text_block(
                canvas, debug_lines,
                org=(40, self.screen_h - 220), scale=0.65, color=(40, 40, 40), line_gap=30,
            )

            draw_camera_preview(canvas, data["frame_bgr"], data["face_found"], self.screen_w)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or self._window_closed():
                return None

            if phase == "capture" and elapsed >= capture_seconds:
                enough_samples = len(valid_samples) >= calib.MIN_VALID_SAMPLES_PER_POINT
                time_exhausted = elapsed >= capture_seconds + calib.MAX_CAPTURE_EXTENSION_S
                if enough_samples or time_exhausted:
                    return calib.choose_point_samples(valid_samples, any_samples)

    # -- estado: calibración -------------------------------------------------
    def _state_calibrating(self):
        canvas_base = self._blank_canvas()
        points_px = calib.points_pct_to_pixels(
            calib.CALIBRATION_POINTS_PCT, self.screen_w, self.screen_h
        )

        features, targets = [], []
        for i, point_px in enumerate(points_px, start=1):
            label = f"Mira el punto rojo  ({i}/9)"
            samples = self._capture_point(
                point_px,
                calib.SECONDS_TO_LOOK_AT_POINT,
                calib.SECONDS_CAPTURING,
                label,
                canvas_base,
            )
            if samples is None:
                return STATE_QUIT, None

            aggregated = calib.aggregate_samples(samples)
            if aggregated is not None:
                features.append(aggregated)
                targets.append(point_px)

        if len(features) < calib.MIN_POINTS_TO_FIT:
            self._show_message(
                [
                    "No se pudo calibrar correctamente.",
                    "Se perdio el rostro durante varios puntos.",
                    "",
                    "Presiona ESPACIO para volver a intentar.",
                    "ESC para salir.",
                ],
                allowed_keys=(32, 27),
            )
            key = self._last_message_key
            if key == 27:
                return STATE_QUIT, None
            return STATE_INSTRUCTIONS, None

        self.mapper.fit(features, targets)
        return STATE_VALIDATING, None

    # -- estado: validación ---------------------------------------------------
    def _state_validating(self):
        canvas_base = self._blank_canvas()
        points_px = calib.points_pct_to_pixels(
            calib.VALIDATION_POINTS_PCT, self.screen_w, self.screen_h
        )

        features, targets = [], []
        for i, point_px in enumerate(points_px, start=1):
            label = f"Validacion: mira el punto  ({i}/{len(points_px)})"
            samples = self._capture_point(
                point_px,
                calib.SECONDS_TO_LOOK_AT_POINT * 0.6,
                calib.SECONDS_CAPTURING * 0.6,
                label,
                canvas_base,
            )
            if samples is None:
                return STATE_QUIT, None

            aggregated = calib.aggregate_samples(samples)
            if aggregated is not None:
                features.append(aggregated)
                targets.append(point_px)

        if len(features) < 3:
            error_pct = 100.0  # no hay suficientes datos: se fuerza a recalibrar
        else:
            error_pct, _ = calib.compute_validation_error_pct(
                self.mapper, features, targets, self.screen_w, self.screen_h
            )
        self.last_error_pct = error_pct

        # MVP: siempre se pasa al seguimiento para ver la grilla 3x3.
        # El error se informa; R permite recalibrar después.
        quality = "aceptable" if error_pct <= calib.MAX_ERROR_PCT else "modesta"
        self._show_message(
            [
                "Calibracion lista.",
                f"Error promedio: {error_pct:.1f}% de la pantalla ({quality})",
                "",
                "Presiona ESPACIO para ver la grilla 3x3.",
                "Durante el seguimiento, R recalibra. ESC sale.",
            ],
            allowed_keys=(32, 27),
            auto_continue_seconds=3.0,
        )
        if self._last_message_key == 27:
            return STATE_QUIT, None
        self.stabilizer.reset()
        self.smoothed_point = None
        return STATE_TRACKING, None

    def _show_message(self, lines, allowed_keys=(32,), auto_continue_seconds=None):
        """Pantalla de mensaje bloqueante hasta que se presione una tecla
        permitida (o se agote `auto_continue_seconds`, si se indica)."""
        t_start = time.time()
        while True:
            canvas = self._blank_canvas()
            draw_text_block(canvas, lines, org=(40, 80), scale=0.9, line_gap=38)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in allowed_keys:
                self._last_message_key = key
                return
            if auto_continue_seconds is not None and (time.time() - t_start) >= auto_continue_seconds:
                self._last_message_key = 32
                return

    # -- estado: seguimiento (modo normal) -----------------------------------
    def _state_tracking(self):
        last_blink_at = None
        was_closed = False
        selected_cell = None
        last_open_cell = None
        while True:
            data = self._read()
            if data is None:
                return STATE_QUIT, "Se perdió la conexión con la cámara."

            fps = self.fps_meter.tick()
            canvas = self.grid_canvas.copy()

            estado_txt = "Buscando rostro..."
            cuadro_txt = "-"
            gaze_txt = "(-, -)"
            gazed_cell = None

            # Señal visual de parpadeo, independiente de si se rechazan
            # frames para el gaze. La predicción de mirada no se toca.
            eyes_closed = detect_blink(
                data.get("blink_score_a", 0.0),
                data.get("blink_score_b", 0.0),
                enabled=True,
            )
            now = time.time()
            if eyes_closed:
                last_blink_at = now
            show_blink_frame = blink_frame_active(now, last_blink_at)

            use_feature = data["face_found"] and data["feature"] is not None
            instant_cell = None
            # Solo se congela el cuadro si AMBOS ojos superan el umbral
            # (ver detect_blink). Un parpadeo real es breve; no se deja
            # de dibujar la última predicción.
            if use_feature:
                raw_x, raw_y = self.mapper.predict(data["feature"])
                raw_x = float(np.clip(raw_x, 0, self.screen_w - 1))
                raw_y = float(np.clip(raw_y, 0, self.screen_h - 1))
                # Casilla de ESTE frame (sin voto mayoritario) para marcar
                # al instante en el flanco de subida del parpadeo.
                instant_cell = compute_cell(raw_x, raw_y, self.screen_w, self.screen_h)

            if use_feature and not data["blink"]:
                if self.smoothed_point is None:
                    self.smoothed_point = (raw_x, raw_y)
                else:
                    sx, sy = self.smoothed_point
                    self.smoothed_point = (
                        SMOOTHING_ALPHA * raw_x + (1 - SMOOTHING_ALPHA) * sx,
                        SMOOTHING_ALPHA * raw_y + (1 - SMOOTHING_ALPHA) * sy,
                    )

                raw_cell = compute_cell(self.smoothed_point[0], self.smoothed_point[1],
                                         self.screen_w, self.screen_h)
                gazed_cell = self.stabilizer.update(raw_cell)
                # Misma casilla que el punto azul (no el voto de 10 frames).
                last_open_cell = raw_cell

                estado_txt = "Deteccion activa"
                cuadro_txt = str(gazed_cell)
                gaze_txt = f"({int(self.smoothed_point[0])}, {int(self.smoothed_point[1])})"
            elif use_feature and data["blink"]:
                if self.smoothed_point is None:
                    self.smoothed_point = (raw_x, raw_y)
                    self.stabilizer.update(instant_cell)
                if self.smoothed_point is not None and self.stabilizer.history:
                    gazed_cell = self.stabilizer.history[-1]
                    cuadro_txt = str(gazed_cell)
                    gaze_txt = f"({int(self.smoothed_point[0])}, {int(self.smoothed_point[1])})"
                estado_txt = "Parpadeo (se mantiene ultima posicion)"
            elif data["face_found"]:
                estado_txt = "Rostro detectado, landmarks incompletos"
            else:
                estado_txt = "No se detecta el rostro"

            mark_cell = gaze_cell_for_blink_mark(last_open_cell, instant_cell, eyes_closed)
            selected_cell = next_selected_cell(
                selected_cell, mark_cell, eyes_closed, was_closed
            )
            was_closed = eyes_closed

            if selected_cell is not None:
                draw_selected_cell(canvas, selected_cell, self.screen_w, self.screen_h)
            draw_cell_labels(canvas, self.screen_w, self.screen_h, selected_cell)

            if self.smoothed_point is not None:
                cv2.circle(canvas, (int(self.smoothed_point[0]), int(self.smoothed_point[1])),
                           10, (255, 0, 0), -1)

            if show_blink_frame:
                draw_blink_frame(canvas)
                estado_txt = "Parpadeo detectado"

            azul_txt = str(selected_cell) if selected_cell is not None else "-"
            hud_lines = [
                f"Gaze: {gaze_txt}",
                f"Mirando: {cuadro_txt}",
                f"Azul: {azul_txt}",
                f"FPS: {fps:.0f}",
                f"Estado: {estado_txt}",
                f"Blink R/L: {data.get('blink_score_a', 0.0):.2f}/"
                f"{data.get('blink_score_b', 0.0):.2f}",
            ]
            draw_text_block(canvas, hud_lines, org=(30, 45), scale=0.75, line_gap=32)
            draw_camera_preview(canvas, data["frame_bgr"], data["face_found"], self.screen_w)

            controles = "ESC: salir   R: recalibrar"
            cv2.putText(canvas, controles, (30, self.screen_h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or self._window_closed():
                return STATE_QUIT, None
            if key in (ord("r"), ord("R")):
                return STATE_INSTRUCTIONS, None

    # -- pantalla de error fatal ----------------------------------------------
    def _show_fatal_error(self, message):
        lines = ["ERROR", "", message, "", "Presiona cualquier tecla para salir."]
        canvas = self._blank_canvas() if hasattr(self, "screen_w") else np.full((720, 1280, 3), 255, np.uint8)
        draw_text_block(canvas, lines, org=(40, 80), scale=0.9, color=(0, 0, 200), line_gap=40)
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(0)

    # -- bucle principal -------------------------------------------------------
    def run(self):
        if self.camera_error:
            self._show_fatal_error(self.camera_error)
            self._cleanup()
            return

        state = STATE_INSTRUCTIONS
        try:
            while state != STATE_QUIT:
                if state == STATE_INSTRUCTIONS:
                    state, err = self._state_instructions()
                elif state == STATE_CALIBRATING:
                    state, err = self._state_calibrating()
                elif state == STATE_VALIDATING:
                    state, err = self._state_validating()
                elif state == STATE_TRACKING:
                    state, err = self._state_tracking()
                else:
                    break

                if err:
                    self._show_fatal_error(err)
                    break
        finally:
            self._cleanup()

    def _cleanup(self):
        if self.tracker is not None:
            self.tracker.release()
        cv2.destroyAllWindows()


def main():
    app = EyeTrackerApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
