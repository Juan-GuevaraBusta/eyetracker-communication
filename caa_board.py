"""
Tablero CAA de cabecera, el mismo árbol que interfaz-caa-pacientes.html.

La mirada cae en un botón. El parpadeo lo activa. OpenCV no dibuja
emojis ni tildes: la frase es el objeto, el acento se aplana al pintar.
"""

import unicodedata

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAPER = (230, 239, 243)  # #f3efe6 en BGR

# bg, borde, tinta, encabezado
CATS = {
    "A": ((251, 228, 212), (148, 74, 14), (71, 35, 12), (252, 240, 231), "Ayuda"),
    "B": ((168, 215, 255), (0, 62, 138), (0, 28, 61), (220, 240, 255), "Sensación"),
    "C": ((198, 232, 197), (34, 92, 26), (20, 51, 14), (229, 246, 228), "Comunicar"),
    "D": ((247, 208, 226), (143, 36, 76), (84, 18, 42), (252, 234, 243), "Hacer o conocer"),
    "H": ((240, 235, 230), (64, 49, 36), (21, 25, 28), (240, 245, 247), ""),
}
YES = ((188, 224, 183), (36, 96, 20), (22, 61, 12))
NO = ((219, 226, 228), (64, 51, 44), (21, 25, 28))
RED = ((22, 27, 142), (13, 16, 92), (247, 247, 247))
RED_SOFT = ((195, 199, 244), (22, 27, 142), (13, 16, 92))
BACK = ((21, 25, 28), (21, 25, 28), (240, 245, 247))
ARROW = ((240, 245, 247), (64, 49, 36), (21, 25, 28))
WHITE = (247, 245, 240)


def _plain(text):
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.replace("¿", "").replace("¡", "").replace("—", "-")


def G(label, to):
    return {"label": label, "to": to}


def F(label, urgent=False, ev=False, un=False):
    return {"label": label, "urgent": urgent, "ev": ev, "un": un}


def P(label, un=False):
    return {"label": label, "intensity": True, "un": un}


EXTRAS = [
    F("Quiero comunicar algo diferente", un=True),
    F("No encuentro lo que necesito", un=True),
    F("Necesito ayuda, pero no sé cómo explicarlo", un=True),
    F("Quiero hablar con el personal de salud", un=True),
    F("Quiero otra cosa", un=True),
]

INTENS = [
    {"label": "Es una molestia leve"},
    {"label": "Es una molestia moderada"},
    {"label": "Es una molestia intensa", "urgent": True},
    {"label": "No sé cómo describirla"},
]


def _nodes():
    nodes = {
        "home": {"title": "¿Qué quieres comunicar?", "cat": "H", "items": [
            G("QUIERO QUE ME AYUDEN", "menuA"),
            G("SIENTO ALGO", "menuB"),
            G("QUIERO COMUNICAR ALGO", "menuC"),
            G("QUIERO HACER O CONOCER ALGO", "menuD"),
        ]},
        "menuA": {"title": "Quiero que me ayuden", "cat": "A", "items": [
            G("Con mi cuerpo y comodidad", "A_cuerpo"),
            G("Con mi alimentación e hidratación", "A_alim"),
            G("Con mi higiene y cuidado personal", "A_hig"),
            G("Con mis necesidades fisiológicas", "A_fis"),
            F("Quiero que revisen mi tubo o equipo respiratorio", ev=True),
            F("Necesito ayuda para realizar una actividad"),
            F("Necesito otra cosa", un=True),
            EXTRAS[2],
        ]},
        "A_cuerpo": {"title": "Con mi cuerpo y comodidad", "cat": "A", "items": [
            F("Necesito que me cambien de posición"),
            F("Necesito que me acomoden la cabeza"),
            F("Necesito que me acomoden los brazos o las piernas"),
            F("Necesito que me ayuden a moverme"),
            F("Necesito descansar o dormir"),
            F("Necesito que me ayuden a vestirme o acomodar mi ropa"),
            F("Necesito que ajusten mi temperatura o mis cobijas"),
            F("Necesito ayuda con otro aspecto de mi comodidad", un=True),
        ]},
        "A_alim": {"title": "Con mi alimentación e hidratación", "cat": "A", "items": [
            F("Necesito que me humedezcan la boca", ev=True),
            F("Necesito que me hidraten los labios", ev=True),
            F("Necesito que me ayuden con mi alimentación", ev=True),
            F("Necesito consultar si puedo beber o comer algo", ev=True),
        ]},
        "A_hig": {"title": "Con mi higiene y cuidado personal", "cat": "A", "items": [
            F("Necesito que me limpien la boca"),
            F("Necesito que me limpien la cara o los ojos"),
            F("Necesito que me limpien el cuerpo"),
            F("Necesito que me acomoden el cabello o me ayuden con mi aseo"),
        ]},
        "A_fis": {"title": "Con mis necesidades fisiológicas", "cat": "A", "items": [
            F("Necesito orinar"),
            F("Necesito evacuar"),
            F("Necesito que revisen mi sonda o bolsa de orina", ev=True),
            F("Necesito ayuda con otra necesidad fisiológica", un=True),
        ]},
        "menuB": {"title": "Siento algo", "cat": "B", "items": [
            G("Siento dolor", "B_dolor"),
            G("Tengo molestias físicas", "B_mol"),
            G("Siento cambios en mi cuerpo", "B_cam"),
            G("Tengo dificultad para respirar", "B_resp"),
            F("Siento que algo puede lastimarme", urgent=True),
            EXTRAS[2], EXTRAS[1], EXTRAS[3],
        ]},
        "B_dolor": {"title": "Siento dolor", "cat": "B", "items": [
            P("Me duele la cabeza"),
            P("Me duele el pecho"),
            P("Me duele el abdomen"),
            G("Me duele otra parte del cuerpo", "B_dolor2"),
        ]},
        "B_dolor2": {"title": "Me duele otra parte del cuerpo", "cat": "B", "items": [
            P("Me duele la espalda"),
            P("Me duelen los brazos o las manos"),
            P("Me duelen las piernas o los pies"),
            P("Me duele otra zona", un=True),
        ]},
        "B_mol": {"title": "Tengo molestias físicas", "cat": "B", "items": [
            P("Tengo náuseas"),
            P("Tengo picazón"),
            P("Tengo mareo"),
            P("Tengo molestias en la boca o garganta"),
        ]},
        "B_cam": {"title": "Siento cambios en mi cuerpo", "cat": "B", "items": [
            F("Tengo frío"),
            F("Tengo calor"),
            F("Tengo escalofríos"),
            F("Siento sudoración excesiva"),
        ]},
        "B_resp": {"title": "Tengo dificultad para respirar", "cat": "B", "items": [
            F("Siento que me falta el aire", urgent=True),
            F("Siento presión en el pecho", urgent=True),
            F("Siento molestias al respirar"),
            F("Siento molestias relacionadas con el tubo", urgent=True),
        ]},
        "menuC": {"title": "Quiero comunicar algo", "cat": "C", "items": [
            G("Quiero expresar cómo me siento", "C_sent"),
            G("Quiero comunicarme con alguien", "C_com"),
            G("Quiero expresar mis preferencias", "C_pref"),
            G("Quiero hablar sobre mis creencias y valores", "C_cre"),
            EXTRAS[0], EXTRAS[1], EXTRAS[3], EXTRAS[4],
        ]},
        "C_sent": {"title": "Quiero expresar cómo me siento", "cat": "C",
                   "pt": {2: "¿Qué necesito emocionalmente?"}, "items": [
            F("Estoy triste"), F("Estoy preocupado/a"), F("Estoy asustado/a"), F("Estoy tranquilo/a"),
            F("Me siento solo/a"), F("Estoy frustrado/a"), F("Estoy aburrido/a"), F("Me siento acompañado/a"),
            F("Necesito que me escuchen"), F("Necesito que me tranquilicen"),
            F("Necesito que me acompañen"), F("Necesito estar un momento a solas"),
        ]},
        "C_com": {"title": "Quiero comunicarme con alguien", "cat": "C", "items": [
            G("Quiero hablar con mi familia", "C_fam"),
            F("Quiero hablar con mis amigos"),
            G("Quiero hablar con el personal de salud", "C_salud"),
            F("Quiero comunicarme con otra persona", un=True),
        ]},
        "C_fam": {"title": "Quiero hablar con mi familia", "cat": "C", "items": [
            F("Los extraño"), F("Quiero saber cómo están"),
            F("Quiero que vengan a visitarme"), F("Quiero enviarles un mensaje"),
        ]},
        "C_salud": {"title": "Quiero hablar con el personal de salud", "cat": "C", "items": [
            F("Quiero hacerles una pregunta"), F("Quiero contarles algo"),
            F("Quiero que me expliquen algo"), F("Quiero agradecerles"),
        ]},
        "C_pref": {"title": "Quiero expresar mis preferencias", "cat": "C", "items": [
            F("Prefiero estar acompañado/a"), F("Prefiero tener privacidad"),
            F("Quiero participar en las decisiones sobre mi cuidado"),
            G("No estoy de acuerdo con algo", "C_desac"),
        ]},
        "C_desac": {"title": "No estoy de acuerdo con algo", "cat": "C", "items": [
            F("No quiero que realicen ese procedimiento todavía"),
            F("Quiero que me expliquen antes de hacerlo"),
            F("Quiero expresar una preocupación"),
            F("Quiero hablar con alguien sobre mi decisión"),
        ]},
        "C_cre": {"title": "Quiero hablar sobre mis creencias y valores", "cat": "C", "items": [
            F("Quiero orar o rezar"), F("Quiero recibir acompañamiento espiritual"),
            F("Quiero hablar sobre mis creencias"),
            F("Quiero contactar a alguien de mi comunidad religiosa o espiritual"),
            F("Quiero que respeten mis costumbres"),
            F("Quiero hablar sobre algo importante para mí"),
        ]},
        "menuD": {"title": "Quiero hacer o conocer algo", "cat": "D", "items": [
            G("Quiero entretenerme", "D_ent"),
            G("Quiero aprender o saber algo", "D_apr"),
            G("Quiero participar en mi recuperación", "D_part"),
            G("Quiero realizar una actividad", "D_act"),
            EXTRAS[0], EXTRAS[1], EXTRAS[3], EXTRAS[4],
        ]},
        "D_ent": {"title": "Quiero entretenerme", "cat": "D", "items": [
            G("Quiero escuchar música", "D_mus"),
            F("Quiero ver televisión o videos"),
            F("Quiero ver fotografías"),
            F("Quiero jugar o realizar una actividad recreativa"),
        ]},
        "D_mus": {"title": "Quiero escuchar música", "cat": "D", "items": [
            F("Quiero escuchar música tranquila"), F("Quiero escuchar música alegre"),
            F("Quiero escuchar mi música favorita"), F("Quiero escuchar otro tipo de música", un=True),
        ]},
        "D_apr": {"title": "Quiero aprender o saber algo", "cat": "D", "items": [
            F("Quiero saber qué está pasando conmigo"), F("Quiero conocer mi diagnóstico"),
            F("Quiero entender mi tratamiento"), F("Quiero saber cómo va mi recuperación"),
            F("Quiero saber qué procedimientos me van a realizar"),
            F("Quiero saber para qué sirven los dispositivos que tengo conectados"),
            F("Quiero saber cuánto tiempo estaré hospitalizado/a"),
            F("Quiero hacer otra pregunta", un=True),
        ]},
        "D_part": {"title": "Quiero participar en mi recuperación", "cat": "D", "items": [
            F("Quiero conocer los avances de mi recuperación"),
            F("Quiero participar en las decisiones sobre mi cuidado"),
            F("Quiero establecer una meta para mi recuperación"),
            F("Quiero colaborar en las actividades que pueda realizar"),
        ]},
        "D_act": {"title": "Quiero realizar una actividad", "cat": "D", "items": [
            F("Quiero leer"), F("Quiero escribir o dibujar"),
            F("Quiero revisar mis fotografías"), F("Quiero proponer otra actividad", un=True),
            F("Quiero hablar sobre mi trabajo o estudios"),
            F("Quiero organizar algo pendiente"),
            F("Quiero realizar una actividad que me haga sentir productivo/a"),
        ]},
    }
    for node in nodes.values():
        items = list(node["items"])
        k = 0
        while len(items) % 4 != 0:
            items.append(EXTRAS[k])
            k += 1
        node["pages"] = [items[i:i + 4] for i in range(0, len(items), 4)]
    return nodes


NODES = _nodes()


class Board:
    def __init__(self):
        self.hist = [{"kind": "menu", "node": "home", "page": 0}]
        self.overlay = None

    def cur(self):
        return self.hist[-1]

    def mark(self):
        screen = self.cur()
        return (self.overlay, len(self.hist), screen.get("kind"), screen.get("node"), screen.get("page"), screen.get("msg"))

    def activate(self, act):
        """Igual que el clic del HTML. Devuelve la frase si acaba de enviarse."""
        if not act:
            return None
        if self.overlay:
            if act == "helpYes":
                self.overlay = "alert"
            elif act in ("helpNo", "helpDone", "back"):
                self.overlay = None
            return "AYUDA INMEDIATA" if act == "helpYes" else None

        screen = self.cur()
        if act == "help":
            self.overlay = "confirm"
            return None
        if act == "prev" and screen["kind"] == "menu" and screen["page"] > 0:
            screen["page"] -= 1
            return None
        if act == "next" and screen["kind"] == "menu":
            pages = NODES[screen["node"]]["pages"]
            if screen["page"] < len(pages) - 1:
                screen["page"] += 1
            return None
        if act in ("back", "no"):
            if len(self.hist) > 1:
                self.hist.pop()
            return None
        if act == "home":
            self.hist = [{"kind": "menu", "node": "home", "page": 0}]
            return None
        if act == "yes":
            self.hist.append(dict(screen, kind="sent"))
            return screen.get("msg")
        if act.startswith("open:"):
            self._open(screen, int(act.split(":", 1)[1]))
        return None

    def _open(self, screen, index):
        if screen["kind"] == "intensity":
            level = INTENS[index]
            self.hist.append({
                "kind": "confirm",
                "msg": screen["base"] + ". " + level["label"] + ".",
                "cat": screen["cat"],
                "urgent": bool(level.get("urgent")),
                "ev": False,
                "un": False,
            })
            return
        node = NODES[screen["node"]]
        item = node["pages"][screen["page"]][index]
        if item.get("to"):
            self.hist.append({"kind": "menu", "node": item["to"], "page": 0})
        elif item.get("intensity"):
            self.hist.append({
                "kind": "intensity", "base": item["label"], "cat": node["cat"],
            })
        else:
            self.hist.append({
                "kind": "confirm",
                "msg": item["label"],
                "cat": node["cat"],
                "urgent": bool(item.get("urgent")),
                "ev": bool(item.get("ev")),
                "un": bool(item.get("un")),
            })


def describe(board):
    """Pantalla actual, sin píxeles. La usa el dibujo y las pruebas."""
    if board.overlay == "confirm":
        return {
            "kind": "help-ask", "cat": "H", "title": "¿Necesitas ayuda inmediata?",
            "kicker": "", "sub": "", "page": "", "help": False, "footer": "back",
            "arrows": False, "can_prev": False, "can_next": False,
            "tiles": [],
            "message": "Se mostrara un aviso de ayuda inmediata",
            "note": "Simulacion del prototipo: no se enviara ninguna alerta real.",
            "band": "", "urgent": False, "alert": False,
            "actions": [("helpYes", "SI, NECESITO AYUDA", "red"), ("helpNo", "NO, CANCELAR", "no")],
        }
    if board.overlay == "alert":
        return {
            "kind": "help-alert", "cat": "H", "title": "Ayuda inmediata",
            "kicker": "", "sub": "", "page": "", "help": False, "footer": None,
            "arrows": False, "can_prev": False, "can_next": False, "tiles": [],
            "message": "El paciente solicita ayuda inmediata",
            "note": "Aviso simulado del prototipo: no se ha enviado ninguna alerta real.",
            "band": "Ayuda inmediata", "urgent": True, "alert": True,
            "actions": [("helpDone", "CONTINUAR", "light")],
        }

    screen = board.cur()
    if screen["kind"] == "menu":
        node = NODES[screen["node"]]
        page = screen["page"]
        home = len(board.hist) == 1
        title = node.get("pt", {}).get(page, node["title"])
        tiles = []
        for item in node["pages"][page]:
            cat = NODES[item["to"]]["cat"] if home else node["cat"]
            tiles.append((item["label"], cat))
        total = len(node["pages"])
        return {
            "kind": "menu", "cat": node["cat"], "title": title,
            "kicker": "" if home else CATS[node["cat"]][4],
            "sub": "", "page": f"{page + 1} / {total}" if total > 1 else "",
            "help": True, "footer": "hint" if home else "back",
            "arrows": total > 1, "can_prev": page > 0, "can_next": page < total - 1,
            "tiles": tiles, "message": "", "note": "", "band": "",
            "urgent": False, "alert": False, "actions": [], "home": home,
        }
    if screen["kind"] == "intensity":
        return {
            "kind": "intensity", "cat": screen["cat"],
            "title": "¿Qué tan intensa es la molestia?",
            "kicker": CATS[screen["cat"]][4], "sub": screen["base"], "page": "",
            "help": True, "footer": "back", "arrows": False,
            "can_prev": False, "can_next": False,
            "tiles": [(item["label"], screen["cat"]) for item in INTENS],
            "message": "", "note": "", "band": "", "urgent": False, "alert": False,
            "actions": [], "home": False,
        }
    if screen["kind"] == "confirm":
        return {
            "kind": "confirm", "cat": screen["cat"],
            "title": "¿Esto es lo que quieres comunicar?",
            "kicker": CATS[screen["cat"]][4], "sub": "", "page": "",
            "help": True, "footer": "back", "arrows": False,
            "can_prev": False, "can_next": False, "tiles": [],
            "message": screen["msg"], "note": "",
            "band": "Mensaje prioritario" if screen.get("urgent") else "",
            "urgent": bool(screen.get("urgent")), "alert": False,
            "actions": [("yes", "SI, ES ESO", "yes"), ("no", "NO, QUIERO CAMBIAR", "no")],
            "home": False,
        }
    notes = []
    if screen.get("ev"):
        notes.append("El personal de salud evaluara esta solicitud antes de actuar.")
    if screen.get("un"):
        notes.append("No esta en la lista: el personal debe acercarse a escuchar.")
    notes.append("Prototipo: comunicacion simulada. No se envio a ningun sistema real.")
    urgent = bool(screen.get("urgent"))
    return {
        "kind": "sent", "cat": screen["cat"],
        "title": "Aviso prioritario" if urgent else "Mensaje comunicado",
        "kicker": CATS[screen["cat"]][4], "sub": "", "page": "",
        "help": False, "footer": None, "arrows": False,
        "can_prev": False, "can_next": False, "tiles": [],
        "message": screen["msg"], "note": " ".join(notes),
        "band": "Aviso prioritario para el personal de salud" if urgent else "Solicitud comunicada al personal de salud",
        "urgent": urgent, "alert": urgent,
        "actions": [("home", "VOLVER AL MENU PRINCIPAL", "tile")],
        "home": False,
    }


def _split_grid(rect, rows):
    x0, y0, x1, y1 = rect
    gap = 10
    inner_w = x1 - x0 - gap
    col_w = inner_w // 2
    weights = rows
    total = sum(weights)
    spans = []
    y = y0
    height = y1 - y0 - gap * (len(weights) - 1)
    for i, weight in enumerate(weights):
        h = int(height * weight / total)
        spans.append((y, y + h))
        y += h + gap
    spans[-1] = (spans[-1][0], y1)
    cols = [(x0, x0 + col_w), (x0 + col_w + gap, x1)]
    return spans, cols


def layout(board, w, h):
    """Botones con rectángulo en píxeles. El hit-test de la mirada usa esto."""
    view = describe(board)
    pad_x, pad_y = int(w * 0.012), int(h * 0.012)
    gap = max(8, int(h * 0.012))
    header_h = int(h * 0.13)
    footer_h = int(h * 0.12) if view["footer"] else 0
    x0, x1 = pad_x, w - pad_x
    header = (x0, pad_y, x1, pad_y + header_h)
    footer = (x0, h - pad_y - footer_h, x1, h - pad_y) if footer_h else None
    main_y0 = header[3] + gap
    main_y1 = (footer[1] - gap) if footer else (h - pad_y)
    side = int(w * 0.10) if view["arrows"] else int(w * 0.012)
    if view["arrows"]:
        left = (x0, main_y0, x0 + side, main_y1)
        right = (x1 - side, main_y0, x1, main_y1)
        grid = (left[2] + gap, main_y0, right[0] - gap, main_y1)
    else:
        left = right = None
        grid = (x0, main_y0, x1, main_y1)

    help_w = int(w * 0.22)
    help_box = (
        x1 - help_w, header[1] + int(header_h * 0.14),
        x1 - int(w * 0.012), header[3] - int(header_h * 0.14),
    )
    preview_w, preview_h = 150, max(48, header_h - int(header_h * 0.28))
    if view["help"]:
        preview = (help_box[0] - preview_w - 12, help_box[1], help_box[0] - 12, help_box[1] + preview_h)
    else:
        preview = (x1 - preview_w - 8, header[1] + 8, x1 - 8, header[1] + 8 + preview_h)

    targets = []
    if view["help"]:
        targets.append({"id": "help", "rect": help_box, "label": "AYUDA INMEDIATA", "face": "help", "on": True})
    if view["arrows"]:
        targets.append({"id": "prev", "rect": left, "label": "ANTERIOR", "face": "arrow", "on": view["can_prev"]})
        targets.append({"id": "next", "rect": right, "label": "SIGUIENTE", "face": "arrow", "on": view["can_next"]})
    if view["footer"] == "back":
        targets.append({"id": "back", "rect": footer, "label": "REGRESAR", "face": "back", "on": True})

    if view["tiles"]:
        spans, cols = _split_grid(grid, (1, 1))
        for i, (label, cat) in enumerate(view["tiles"]):
            row, col = divmod(i, 2)
            rect = (cols[col][0], spans[row][0], cols[col][1], spans[row][1])
            targets.append({
                "id": f"open:{i}", "rect": rect, "label": label,
                "face": "tile", "cat": cat, "on": True,
            })
    elif view["actions"]:
        top_w, bot_w = (1.35, 1.0) if view["kind"] != "sent" and view["kind"] != "help-alert" else (1.75, 0.9)
        if view["kind"] in ("sent", "help-alert"):
            top_w, bot_w = 1.75, 0.9
        spans, cols = _split_grid(grid, (top_w, bot_w))
        targets.append({
            "id": "msg", "rect": (grid[0], spans[0][0], grid[2], spans[0][1]),
            "label": view["message"], "face": "msg", "on": False,
        })
        actions = view["actions"]
        if len(actions) == 1:
            targets.append({
                "id": actions[0][0], "rect": (grid[0], spans[1][0], grid[2], spans[1][1]),
                "label": actions[0][1], "face": actions[0][2], "on": True,
            })
        else:
            for i, (act, label, face) in enumerate(actions):
                targets.append({
                    "id": act,
                    "rect": (cols[i][0], spans[1][0], cols[i][1], spans[1][1]),
                    "label": label, "face": face, "on": True,
                })

    return {
        "view": view, "targets": targets, "header": header, "footer": footer,
        "preview": preview, "grid": grid,
    }


def target_at(targets, x, y):
    for target in targets:
        if not target["on"]:
            continue
        x0, y0, x1, y1 = target["rect"]
        if x0 <= x < x1 and y0 <= y < y1:
            return target["id"]
    return None


def _wrap(text, scale, thickness, max_w):
    words = _plain(text).split()
    lines, cur = [], ""
    for word in words:
        trial = word if not cur else cur + " " + word
        tw = cv2.getTextSize(trial, FONT, scale, thickness)[0][0]
        if tw <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def _fit(text, max_w, max_h, scale, min_scale=0.45):
    thickness = max(1, int(round(scale * 2)))
    while scale >= min_scale:
        thickness = max(1, int(round(scale * 2)))
        lines = _wrap(text, scale, thickness, max_w)
        sizes = [cv2.getTextSize(line, FONT, scale, thickness)[0] for line in lines]
        gap = int(max(4, scale * 8))
        total = sum(s[1] for s in sizes) + gap * (len(lines) - 1)
        if sizes and max(s[0] for s in sizes) <= max_w and total <= max_h:
            return lines, scale, thickness, sizes, gap, total
        scale -= 0.05
    lines = _wrap(text, min_scale, 1, max_w)
    sizes = [cv2.getTextSize(line, FONT, min_scale, 1)[0] for line in lines]
    gap = 4
    total = sum(s[1] for s in sizes) + gap * (len(lines) - 1)
    return lines, min_scale, 1, sizes, gap, total


def _text_in(canvas, text, rect, color, scale, align="left", pad=16):
    x0, y0, x1, y1 = rect
    max_w = max(20, x1 - x0 - pad * 2)
    max_h = max(20, y1 - y0 - pad)
    lines, scale, thickness, sizes, gap, total = _fit(text, max_w, max_h, scale)
    y = y0 + max(pad // 2, (y1 - y0 - total) // 2) + sizes[0][1]
    for line, (tw, th) in zip(lines, sizes):
        if align == "center":
            x = x0 + (x1 - x0 - tw) // 2
        else:
            x = x0 + pad
        cv2.putText(canvas, line, (int(x), int(y)), FONT, scale, color, thickness, cv2.LINE_AA)
        y += th + gap


def _box(canvas, rect, fill, border, bar=0):
    x0, y0, x1, y1 = [int(v) for v in rect]
    cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, 4)
    if bar:
        cv2.rectangle(canvas, (x0, y0), (x0 + bar, y1), border, -1)


def _face(view, target):
    kind = target["face"]
    cat = CATS[target.get("cat", view["cat"])]
    if kind == "tile":
        return cat[0], cat[1], cat[2], 14
    if kind == "help":
        return RED[0], RED[1], RED[2], 0
    if kind == "arrow":
        return ARROW[0], ARROW[1], ARROW[2], 0
    if kind == "back":
        return BACK[0], BACK[1], BACK[2], 0
    if kind == "yes":
        return YES[0], YES[1], YES[2], 0
    if kind == "no":
        return NO[0], NO[1], NO[2], 0
    if kind == "red":
        return RED_SOFT[0], RED_SOFT[1], RED_SOFT[2], 0
    if kind == "light":
        return WHITE, RED[1], RED[1], 0
    if view["alert"]:
        return RED[0], RED[1], WHITE, 0
    fill = (244, 246, 255) if view["urgent"] else (240, 245, 247)
    return fill, cat[1], cat[2], 0


def paint(canvas, spec, gazed_id=None):
    view = spec["view"]
    cat = CATS[view["cat"]]
    canvas[:] = PAPER
    hx0, hy0, hx1, hy1 = spec["header"]
    header_fill = RED[0] if view["alert"] else cat[3]
    header_bd = RED[1] if view["alert"] else cat[1]
    ink = WHITE if view["alert"] else cat[2]
    _box(canvas, spec["header"], header_fill, header_bd, bar=16)
    title_right = spec["preview"][0] - 8
    kicker_h = 0
    if view["kicker"] and not view["alert"]:
        cv2.putText(
            canvas, _plain(view["kicker"]).upper(), (hx0 + 28, hy0 + 28),
            FONT, 0.55, cat[1], 2, cv2.LINE_AA,
        )
        kicker_h = 22
    title_bottom = hy1 - (32 if view["sub"] else 6)
    _text_in(
        canvas, view["title"],
        (hx0 + 18, hy0 + 8 + kicker_h, title_right, title_bottom),
        ink, 1.15, align="left", pad=8,
    )
    if view["sub"]:
        cv2.putText(
            canvas, _plain(view["sub"]), (hx0 + 28, hy1 - 12),
            FONT, 0.6, ink, 1, cv2.LINE_AA,
        )
    if view["page"]:
        cv2.putText(
            canvas, view["page"], (title_right - 90, hy0 + 36),
            FONT, 0.8, ink, 2, cv2.LINE_AA,
        )

    for target in spec["targets"]:
        fill, border, color, bar = _face(view, target)
        if not target["on"] and target["face"] == "arrow":
            fill = PAPER
        _box(canvas, target["rect"], fill, border, bar=bar if target["on"] else 0)
        if target["id"] == gazed_id and target["on"]:
            x0, y0, x1, y1 = [int(v) for v in target["rect"]]
            cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x1 - 6, y1 - 6), (20, 20, 20), 6)
        if target["face"] == "msg":
            _paint_message(canvas, view, target["rect"])
            continue
        align = "center" if target["face"] != "tile" else "left"
        label = target["label"]
        if target["face"] == "arrow":
            label = ("<" if target["id"] == "prev" else ">") + "  " + label
        _text_in(canvas, label, target["rect"], color, 0.9 if target["face"] == "tile" else 1.05, align=align)
        if not target["on"] and target["face"] == "arrow":
            x0, y0, x1, y1 = [int(v) for v in target["rect"]]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (160, 160, 160), 2)

    if view["footer"] == "hint" and spec["footer"]:
        _text_in(canvas, "Elige una de las cuatro opciones", spec["footer"], (48, 56, 61), 0.8, align="center")


def _paint_message(canvas, view, rect):
    x0, y0, x1, y1 = [int(v) for v in rect]
    color = WHITE if view["alert"] else CATS[view["cat"]][2]
    y = y0
    if view["band"]:
        band_h = max(36, int((y1 - y0) * 0.16))
        band_bg = WHITE if view["alert"] else ((22, 27, 142) if view["urgent"] else (36, 96, 20))
        band_fg = RED[0] if view["alert"] else WHITE
        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x1 - 4, y0 + band_h), band_bg, -1)
        _text_in(canvas, view["band"], (x0, y0, x1, y0 + band_h), band_fg, 0.7, align="center", pad=8)
        y = y0 + band_h
    body = (x0 + 12, y + 8, x1 - 12, y1 - 8)
    if view["kind"] in ("sent", "help-alert"):
        who_h = 36
        _text_in(canvas, "El paciente comunica", (body[0], body[1], body[2], body[1] + who_h), color, 0.55, align="center", pad=4)
        _text_in(canvas, view["message"], (body[0], body[1] + who_h, body[2], body[3] - 48), color, 1.2, align="center", pad=8)
        _text_in(canvas, view["note"], (body[0], body[3] - 52, body[2], body[3]), color, 0.5, align="center", pad=6)
    else:
        _text_in(canvas, view["message"], body, color, 1.15, align="center")
        if view["note"]:
            _text_in(canvas, view["note"], (body[0], body[3] - 40, body[2], body[3]), color, 0.5, align="center", pad=4)


def render_board(canvas, board, gazed_id=None):
    spec = layout(board, canvas.shape[1], canvas.shape[0])
    paint(canvas, spec, gazed_id)
    return spec
