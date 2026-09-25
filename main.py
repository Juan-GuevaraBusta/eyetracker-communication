"""
main.py
-------
Punto de entrada del eye tracker académico.

Navegación modular 3x3 adaptada a las 14 Necesidades BPS
de Virginia Henderson (categorías clínicas y opciones contextuales).
"""

import sys
import time
import unicodedata
from collections import deque, Counter

import cv2
import numpy as np

import calibration as calib

# ---------------------------------------------------------------------------
# Constantes de la interfaz
# ---------------------------------------------------------------------------
WINDOW_NAME = "Eye Tracker 3x3"

GRID_LINE_THICKNESS = 4
GRID_LINE_COLOR = (200, 210, 220)    # Borde de tarjetas
SELECTED_FILL_COLOR = (235, 150, 40) # Azul BGR: cuadro elegido al parpadear
CARD_BG_COLOR = (255, 255, 255)      # Fondo de las tarjetas

# ---------------------------------------------------------------------------
# Diccionario clínico modular (14 necesidades de Henderson, agrupadas en 9)
# ---------------------------------------------------------------------------
CATEGORIES = {
    1: {"label": "FISIOLOGÍA Y SÍNTOMAS", "icon": "PHYS"},
    2: {"label": "AFECTO Y FAMILIA", "icon": "FAMILY"},
    3: {"label": "ESPIRITUALIDAD", "icon": "FAITH"},
    4: {"label": "INFORMACIÓN MÉDICA", "icon": "INFO"},
    5: {"label": "RESPUESTAS RÁPIDAS", "icon": "QUICK"},
    6: {"label": "CONFORT Y ENTORNO", "icon": "COMFORT"},
    7: {"label": "ESTADO EMOCIONAL", "icon": "EMOTION"},
    8: {"label": "DISPOSITIVOS Y APOYO", "icon": "DEVICES"},
    9: {"label": "AYUDA URGENTE", "icon": "SOS"},
}

SUBMENUS = {
    1: {  # Fisiología y síntomas
        1: {"label": "Tengo sed", "icon": "WATER"},
        2: {"label": "Tengo hambre", "icon": "FOOD"},
        3: {"label": "Dolor de pecho", "icon": "PAIN"},
        4: {"label": "Dolor de garganta", "icon": "PAIN"},
        5: {"label": "Siento náuseas", "icon": "SICK"},
        6: {"label": "Me pica / escozor", "icon": "PAIN"},
        7: {"label": "Me cuesta respirar", "icon": "SOS"},
        8: {"label": "Tengo frío / calor", "icon": "COMFORT"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
    2: {  # Afecto y familia
        1: {"label": "Llamen a mi familia", "icon": "FAMILY"},
        2: {"label": "Quiero verlos", "icon": "FAMILY"},
        3: {"label": "Estoy preocupado", "icon": "EMOTION"},
        4: {"label": "Los extraño", "icon": "EMOTION"},
        5: {"label": "¿Cómo están?", "icon": "CHAT"},
        6: {"label": "Abrazo / Mano", "icon": "FAMILY"},
        7: {"label": "Videollamada", "icon": "CHAT"},
        8: {"label": "Necesito acompañamiento", "icon": "FAMILY"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
    3: {  # Espiritualidad
        1: {"label": "Quiero rezar", "icon": "FAITH"},
        2: {"label": "Llamar sacerdote", "icon": "FAITH"},
        3: {"label": "Llamar pastor", "icon": "FAITH"},
        4: {"label": "Quiero una Biblia", "icon": "FAITH"},
        5: {"label": "Quiero un Rosario", "icon": "FAITH"},
        6: {"label": "Estoy sufriendo", "icon": "EMOTION"},
        7: {"label": "Busco paz", "icon": "FAITH"},
        8: {"label": "Oración en familia", "icon": "FAMILY"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
    4: {  # Información médica
        1: {"label": "¿Mejorará mi salud?", "icon": "INFO"},
        2: {"label": "Quiero informe médico", "icon": "INFO"},
        3: {"label": "¿Es de mañana/noche?", "icon": "INFO"},
        4: {"label": "¿Qué procedimiento harán?", "icon": "INFO"},
        5: {"label": "¿Riesgo de complicación?", "icon": "INFO"},
        6: {"label": "¿Cuánto tiempo llevo aquí?", "icon": "INFO"},
        7: {"label": "Hablar con el doctor", "icon": "INFO"},
        8: {"label": "Explicar tratamiento", "icon": "INFO"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
    5: {  # Respuestas rápidas
        1: {"label": "SÍ", "icon": "YES"},
        2: {"label": "NO", "icon": "NO"},
        3: {"label": "Tal vez", "icon": "CHAT"},
        4: {"label": "Gracias", "icon": "CHAT"},
        5: {"label": "No entiendo", "icon": "INFO"},
        6: {"label": "Por favor", "icon": "CHAT"},
        7: {"label": "Espere un momento", "icon": "QUICK"},
        8: {"label": "Está bien", "icon": "YES"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
    6: {  # Confort y entorno
        1: {"label": "Cambiar de posición", "icon": "COMFORT"},
        2: {"label": "Limpiar/Aseo", "icon": "COMFORT"},
        3: {"label": "Apagar/Prender luz", "icon": "COMFORT"},
        4: {"label": "Ajustar cama", "icon": "COMFORT"},
        5: {"label": "Mucho ruido", "icon": "COMFORT"},
        6: {"label": "Acomodar almohada", "icon": "COMFORT"},
        7: {"label": "Tapar con cobija", "icon": "COMFORT"},
        8: {"label": "Quitar cobija", "icon": "COMFORT"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
    7: {  # Estado emocional
        1: {"label": "Tengo miedo", "icon": "EMOTION"},
        2: {"label": "Estoy enojado", "icon": "EMOTION"},
        3: {"label": "Me siento triste", "icon": "EMOTION"},
        4: {"label": "Estoy tranquilo", "icon": "EMOTION"},
        5: {"label": "Me siento solo", "icon": "EMOTION"},
        6: {"label": "Siento ansiedad", "icon": "EMOTION"},
        7: {"label": "Tengo esperanza", "icon": "EMOTION"},
        8: {"label": "Necesito ánimo", "icon": "EMOTION"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
    8: {  # Dispositivos y apoyo
        1: {"label": "Gafas", "icon": "DEVICES"},
        2: {"label": "Audífonos", "icon": "DEVICES"},
        3: {"label": "Ajustar tubo/sonda", "icon": "DEVICES"},
        4: {"label": "Succión de flemas", "icon": "DEVICES"},
        5: {"label": "Mano limpia", "icon": "DEVICES"},
        6: {"label": "Retirar sensor", "icon": "DEVICES"},
        7: {"label": "Revisar suero", "icon": "DEVICES"},
        8: {"label": "Timbre / Alarma", "icon": "SOS"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
    9: {  # Ayuda urgente
        1: {"label": "¡ALERTA ENFERMERÍA!", "icon": "SOS"},
        2: {"label": "¡FALTA EL AIRE!", "icon": "SOS"},
        3: {"label": "¡MUCHO DOLOR!", "icon": "SOS"},
        4: {"label": "Me siento mal", "icon": "SOS"},
        5: {"label": "Emergencia", "icon": "SOS"},
        6: {"label": "Asfixia", "icon": "SOS"},
        7: {"label": "Caída / Movimiento", "icon": "SOS"},
        8: {"label": "Llamada inmediata", "icon": "SOS"},
        9: {"label": "<-- VOLVER", "icon": "BACK"},
    },
}

CELL_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

STABILITY_WINDOW = 10                # últimas N predicciones
SMOOTHING_ALPHA = 0.35               # suavizado exponencial del punto de mirada

PREVIEW_W, PREVIEW_H = 220, 165      # miniatura de la cámara en pantalla

# Marco negro alrededor de la grilla al parpadear.
BLINK_FRAME_HOLD_S = 0.35
BLINK_FRAME_MARGIN = 56

# Estados de la máquina de estados
STATE_INSTRUCTIONS = "instructions"
STATE_CALIBRATING = "calibrating"
STATE_VALIDATING = "validating"
STATE_TRACKING = "tracking"
STATE_QUIT = "quit"


def active_items(active_category):
    """Menú principal o submenú de la categoría abierta."""
    if active_category is None:
        return CATEGORIES
    return SUBMENUS[active_category]


def apply_menu_selection(active_category, selected_cell, prev_selected, blink_edge):
    """
    En el flanco de subida del parpadeo:
      - menú principal: entra a esa categoría y limpia la selección;
      - casilla 9 de un submenú: vuelve al menú principal;
      - cualquier otra casilla: confirma la frase (también si se repite).

    Devuelve (categoría activa, casilla seleccionada, frase o None).
    """
    if not blink_edge or selected_cell is None:
        return active_category, selected_cell, None

    if active_category is None:
        if selected_cell == prev_selected or selected_cell not in CATEGORIES:
            return active_category, selected_cell, None
        return selected_cell, None, None

    item = SUBMENUS.get(active_category, {}).get(selected_cell)
    if item is None:
        return active_category, selected_cell, None
    if selected_cell == 9:
        return None, None, None
    return active_category, selected_cell, item["label"]


# ---------------------------------------------------------------------------
# Utilidades de pantalla completa
# ---------------------------------------------------------------------------
def get_screen_resolution():
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

    return 1280, 720


def opencv_text(text):
    """
    Hershey (cv2.putText) no dibuja tildes ni signos invertidos.
    El diccionario conserva el español; en pantalla se aplana.
    """
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return (
        stripped.replace("¿", "")
        .replace("¡", "")
        .replace("—", "-")
    )


# ---------------------------------------------------------------------------
# Dibujo de la interfaz y pictogramas
# ---------------------------------------------------------------------------
def draw_vector_icon(canvas, icon_type, center, color, size=24):
    """Dibuja pictogramas esquemáticos para cada tipo de necesidad."""
    cx, cy = center
    s = size
    if icon_type == "WATER":
        cv2.circle(canvas, (cx, cy + 4), s // 2, color, 2)
        pts = np.array([[cx, cy - s], [cx - s // 2, cy], [cx + s // 2, cy]], np.int32)
        cv2.polylines(canvas, [pts], True, color, 2)
    elif icon_type == "FOOD":
        cv2.rectangle(canvas, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), color, 2)
    elif icon_type in ("PAIN", "SOS"):
        cv2.line(canvas, (cx - s, cy), (cx + s, cy), color, 3)
        cv2.line(canvas, (cx, cy - s), (cx, cy + s), color, 3)
        cv2.circle(canvas, (cx, cy), s + 2, color, 2)
    elif icon_type == "YES":
        pts = np.array([[cx - 12, cy], [cx - 3, cy + 10], [cx + 12, cy - 10]], np.int32)
        cv2.polylines(canvas, [pts], False, color, 3)
    elif icon_type == "NO":
        cv2.line(canvas, (cx - 12, cy - 12), (cx + 12, cy + 12), color, 3)
        cv2.line(canvas, (cx + 12, cy - 12), (cx - 12, cy + 12), color, 3)
    elif icon_type == "BACK":
        pts = np.array([[cx + 10, cy - 10], [cx - 10, cy], [cx + 10, cy + 10]], np.int32)
        cv2.polylines(canvas, [pts], False, color, 3)
    elif icon_type == "FAMILY":
        cv2.circle(canvas, (cx - 8, cy - 6), 6, color, 2)
        cv2.circle(canvas, (cx + 8, cy - 6), 6, color, 2)
        cv2.circle(canvas, (cx, cy + 8), 8, color, 2)
    elif icon_type == "FAITH":
        cv2.line(canvas, (cx, cy - s), (cx, cy + s), color, 2)
        cv2.line(canvas, (cx - s // 2, cy - s // 3), (cx + s // 2, cy - s // 3), color, 2)
    elif icon_type == "PHYS":
        pts = np.array([
            [cx - s, cy], [cx - s // 2, cy], [cx - s // 4, cy - s // 2],
            [cx, cy + s // 2], [cx + s // 3, cy - s // 3], [cx + s, cy],
        ], np.int32)
        cv2.polylines(canvas, [pts], False, color, 2)
    elif icon_type == "INFO":
        cv2.circle(canvas, (cx, cy), s // 2 + 2, color, 2)
        cv2.circle(canvas, (cx, cy - s // 3), 2, color, -1)
        cv2.line(canvas, (cx, cy - s // 6), (cx, cy + s // 3), color, 2)
    elif icon_type == "EMOTION":
        cv2.circle(canvas, (cx, cy), s // 2 + 2, color, 2)
        cv2.circle(canvas, (cx - 5, cy - 3), 2, color, -1)
        cv2.circle(canvas, (cx + 5, cy - 3), 2, color, -1)
        cv2.ellipse(canvas, (cx, cy + 4), (6, 4), 0, 20, 160, color, 2)
    elif icon_type == "CHAT":
        cv2.rectangle(canvas, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 4), color, 2)
        pts = np.array([[cx - 4, cy + s // 4], [cx - 8, cy + s // 2], [cx + 4, cy + s // 4]], np.int32)
        cv2.polylines(canvas, [pts], True, color, 2)
    elif icon_type == "COMFORT":
        cv2.rectangle(canvas, (cx - s // 2, cy - 2), (cx + s // 2, cy + s // 2), color, 2)
        cv2.rectangle(canvas, (cx - s // 2, cy - s // 2), (cx - s // 6, cy), color, 2)
    elif icon_type in ("DEVICES", "QUICK", "SICK"):
        cv2.rectangle(canvas, (cx - s // 2, cy - s // 3), (cx + s // 2, cy + s // 3), color, 2)
        if icon_type == "QUICK":
            cv2.line(canvas, (cx - 4, cy - s // 3), (cx + 2, cy), color, 2)
            cv2.line(canvas, (cx + 2, cy), (cx - 2, cy + s // 3), color, 2)
        elif icon_type == "SICK":
            cv2.line(canvas, (cx - 6, cy), (cx + 6, cy), color, 2)
    else:
        cv2.circle(canvas, (cx, cy), max(4, s // 2), color, 2)


def build_grid_canvas(w, h):
    """Prerenderiza el lienzo base claro una sola vez."""
    return np.full((h, w, 3), 245, dtype=np.uint8)


def cell_bounds(cell_idx, w, h):
    """cell_idx en 1..9 -> (x0, y0, x1, y1) con margen de tarjeta."""
    row = (cell_idx - 1) // 3
    col = (cell_idx - 1) % 3
    x0 = int(col * w / 3) + 8
    x1 = int((col + 1) * w / 3) - 8
    y0 = int(row * h / 3) + 8
    y1 = int((row + 1) * h / 3) - 8
    return x0, y0, x1, y1


def compute_cell(x, y, w, h):
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
    Al parpadear el iris se deforma. Se marca la última casilla
    vista con los ojos abiertos.
    """
    if eyes_closed:
        return last_open
    return instant


def draw_selected_cell(canvas, cell_idx, w, h):
    """Rellena de azul la tarjeta elegida."""
    if cell_idx is None:
        return
    x0, y0, x1, y1 = cell_bounds(cell_idx, w, h)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), SELECTED_FILL_COLOR, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), GRID_LINE_COLOR, GRID_LINE_THICKNESS)


def label_layout_for_cell(text, cell_w, cell_h, max_scale=4.0):
    """
    Parte la frase en líneas y elige la escala más grande que cabe
    en ~86% del ancho y ~62% del alto.
    """
    plain = opencv_text(text)
    lines = plain.split() if " " in plain else [plain]
    max_w = cell_w * 0.86
    max_h = cell_h * 0.62
    scale = float(max_scale)
    thickness = 2
    while scale >= 0.35:
        thickness = max(1, int(round(scale * 2.2)))
        widths, heights = [], []
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, CELL_LABEL_FONT, scale, thickness)
            widths.append(tw)
            heights.append(th)
        gap = int(max(4, scale * 10))
        total_h = sum(heights) + gap * (len(lines) - 1)
        if widths and max(widths) <= max_w and total_h <= max_h:
            return lines, float(scale), int(thickness)
        scale -= 0.05
    thickness = max(1, int(round(0.35 * 2.2)))
    return lines, 0.35, thickness


def draw_cell_labels(canvas, w, h, items_dict, selected_cell=None):
    """Dibuja las tarjetas 3x3 de la categoría activa."""
    for idx, item in items_dict.items():
        x0, y0, x1, y1 = cell_bounds(idx, w, h)
        cw, ch = x1 - x0, y1 - y0
        is_selected = idx == selected_cell

        if not is_selected:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), CARD_BG_COLOR, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), GRID_LINE_COLOR, GRID_LINE_THICKNESS)

        text_color = (255, 255, 255) if is_selected else (30, 40, 50)
        icon_color = (255, 255, 255) if is_selected else (220, 120, 30)

        cx = (x0 + x1) // 2
        cy_icon = y0 + int(ch * 0.30)
        draw_vector_icon(canvas, item["icon"], (cx, cy_icon), icon_color, size=max(16, ch // 10))

        text_top = y0 + int(ch * 0.50)
        text_h = max(20, y1 - text_top - 10)
        lines, scale, thickness = label_layout_for_cell(
            item["label"], cw - 12, text_h, max_scale=0.95
        )
        sizes = [cv2.getTextSize(line, CELL_LABEL_FONT, scale, thickness)[0] for line in lines]
        gap = int(max(4, scale * 10))
        total_h = sum(s[1] for s in sizes) + gap * (len(lines) - 1)
        y = text_top + max(0, (text_h - total_h) // 2) + sizes[0][1]
        for line, (tw, th) in zip(lines, sizes):
            tx = cx - tw // 2
            cv2.putText(
                canvas, line, (tx, y), CELL_LABEL_FONT, scale, text_color, thickness, cv2.LINE_AA
            )
            y += th + gap


def blink_frame_active(now, last_blink_at, hold_s=BLINK_FRAME_HOLD_S):
    if last_blink_at is None:
        return False
    return (now - last_blink_at) <= hold_s


def draw_blink_frame(canvas, margin=BLINK_FRAME_MARGIN):
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
    shown = [opencv_text(line) for line in lines]
    if bg and shown:
        widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for t in shown]
        box_w = max(widths) + 24
        box_h = line_gap * len(shown) + 16
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x - 12, y - 30), (x - 12 + box_w, y - 30 + box_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, dst=canvas)
    for line in shown:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_gap


def draw_camera_preview(canvas, frame_bgr, face_found, w):
    if frame_bgr is None:
        return
    thumb = cv2.resize(frame_bgr, (PREVIEW_W, PREVIEW_H))
    x0 = w - PREVIEW_W - 20
    y0 = 20
    y1 = min(canvas.shape[0], y0 + PREVIEW_H)
    x1 = min(canvas.shape[1], x0 + PREVIEW_W)
    if y0 >= canvas.shape[0] or x0 >= canvas.shape[1]:
        return
    canvas[y0:y1, x0:x1] = thumb[: y1 - y0, : x1 - x0]
    border_color = (60, 180, 60) if face_found else (0, 0, 220)
    cv2.rectangle(canvas, (x0, y0), (x0 + PREVIEW_W, y0 + PREVIEW_H), border_color, 3)
    label = "Rostro OK" if face_found else "Sin rostro"
    cv2.putText(canvas, label, (x0, min(canvas.shape[0] - 8, y0 + PREVIEW_H + 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2, cv2.LINE_AA)


def render_menu_canvas(w, h, active_category=None, selected_cell=None, phrase=None):
    """Lienzo de la grilla, sin cámara. Sirve para revisar el tablero."""
    canvas = build_grid_canvas(w, h)
    items = active_items(active_category)
    if selected_cell is not None:
        draw_selected_cell(canvas, selected_cell, w, h)
    draw_cell_labels(canvas, w, h, items, selected_cell)
    title = "MENU PRINCIPAL" if active_category is None else CATEGORIES[active_category]["label"]
    lines = [f"Categoria: {title}"]
    if phrase:
        lines.append(f"Mensaje: {phrase}")
    draw_text_block(canvas, lines, org=(24, 40), scale=0.7, line_gap=28)
    return canvas


# ---------------------------------------------------------------------------
# Estabilización y control
# ---------------------------------------------------------------------------
class Stabilizer:
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

        # En macOS, abrir fullscreen antes de la cámara puede colgar VideoCapture.
        self.tracker = None
        self.camera_error = None
        try:
            from gaze import GazeTracker
            self.tracker = GazeTracker()
        except Exception as exc:
            self.camera_error = str(exc)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.mapper = calib.LinearGazeMapper()
        self.stabilizer = Stabilizer()
        self.smoothed_point = None
        self.fps_meter = FpsMeter()
        self.last_error_pct = None
        self.last_phrase = None

    def _read(self):
        result = self.tracker.read()
        if not result["ok"]:
            return None
        return result

    def _blank_canvas(self):
        return np.full((self.screen_h, self.screen_w, 3), 255, dtype=np.uint8)

    def _window_closed(self):
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _state_instructions(self):
        instrucciones = [
            "CALIBRACION DE COMUNICADOR UCI",
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
            if key == 27 or self._window_closed():
                return STATE_QUIT, None
            if key == 32 and data["face_found"]:
                return STATE_CALIBRATING, None

    def _capture_point(self, point_px, look_seconds, capture_seconds, label, canvas_base):
        px, py = int(point_px[0]), int(point_px[1])
        t_start = time.time()
        phase = "look"
        valid_samples = []
        any_samples = []

        while True:
            data = self._read()
            if data is None:
                return None

            elapsed = time.time() - t_start
            if phase == "look" and elapsed >= look_seconds:
                phase = "capture"
                t_start = time.time()
                elapsed = 0.0

            canvas = canvas_base.copy()
            draw_text_block(canvas, [label], org=(40, 60), scale=0.9, line_gap=36)

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
            error_pct = 100.0
        else:
            error_pct, _ = calib.compute_validation_error_pct(
                self.mapper, features, targets, self.screen_w, self.screen_h
            )
        self.last_error_pct = error_pct

        quality = "aceptable" if error_pct <= calib.MAX_ERROR_PCT else "modesta"
        self._show_message(
            [
                "Calibracion lista.",
                f"Error promedio: {error_pct:.1f}% de la pantalla ({quality})",
                "",
                "Presiona ESPACIO para ver la grilla de necesidades.",
                "Durante el seguimiento, R recalibra. ESC sale.",
            ],
            allowed_keys=(32, 27),
            auto_continue_seconds=3.0,
        )
        if self._last_message_key == 27:
            return STATE_QUIT, None
        self.stabilizer.reset()
        self.smoothed_point = None
        self.last_phrase = None
        return STATE_TRACKING, None

    def _show_message(self, lines, allowed_keys=(32,), auto_continue_seconds=None):
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

    def _state_tracking(self):
        last_blink_at = None
        was_closed = False
        selected_cell = None
        last_open_cell = None
        active_category = None

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

            from gaze import detect_blink
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

            if use_feature:
                raw_x, raw_y = self.mapper.predict(data["feature"])
                raw_x = float(np.clip(raw_x, 0, self.screen_w - 1))
                raw_y = float(np.clip(raw_y, 0, self.screen_h - 1))
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

                raw_cell = compute_cell(
                    self.smoothed_point[0], self.smoothed_point[1],
                    self.screen_w, self.screen_h,
                )
                gazed_cell = self.stabilizer.update(raw_cell)
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
            prev_selected = selected_cell
            selected_cell = next_selected_cell(
                selected_cell, mark_cell, eyes_closed, was_closed
            )
            blink_edge = eyes_closed and not was_closed
            active_category, selected_cell, phrase = apply_menu_selection(
                active_category, selected_cell, prev_selected, blink_edge
            )
            if phrase:
                self.last_phrase = phrase
                print(f"[seleccion] {phrase}", flush=True)
            was_closed = eyes_closed

            items_to_draw = active_items(active_category)
            if selected_cell is not None:
                draw_selected_cell(canvas, selected_cell, self.screen_w, self.screen_h)
            draw_cell_labels(canvas, self.screen_w, self.screen_h, items_to_draw, selected_cell)

            if self.smoothed_point is not None:
                cv2.circle(
                    canvas,
                    (int(self.smoothed_point[0]), int(self.smoothed_point[1])),
                    10, (255, 0, 0), -1,
                )

            if show_blink_frame:
                draw_blink_frame(canvas)
                estado_txt = "Parpadeo detectado"

            categoria_str = (
                "MENU PRINCIPAL" if active_category is None
                else CATEGORIES[active_category]["label"]
            )
            hud_lines = [categoria_str]
            if self.last_phrase:
                hud_lines.append(self.last_phrase)
            draw_text_block(canvas, hud_lines, org=(30, 42), scale=0.7, line_gap=28)
            draw_camera_preview(canvas, data["frame_bgr"], data["face_found"], self.screen_w)

            controles = (
                f"Mirando {cuadro_txt}   {estado_txt}   {fps:.0f} fps"
                "    ESC salir   R recalibrar   Parpadeo elige"
            )
            cv2.putText(
                canvas, opencv_text(controles), (30, self.screen_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or self._window_closed():
                return STATE_QUIT, None
            if key in (ord("r"), ord("R")):
                return STATE_INSTRUCTIONS, None

    def _show_fatal_error(self, message):
        lines = ["ERROR", "", message, "", "Presiona cualquier tecla para salir."]
        canvas = self._blank_canvas() if hasattr(self, "screen_w") else np.full((720, 1280, 3), 255, np.uint8)
        draw_text_block(canvas, lines, org=(40, 80), scale=0.9, color=(0, 0, 200), line_gap=40)
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(0)

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
