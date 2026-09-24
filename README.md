# Eye Tracker Communication

Eye tracker académico con webcam: detecta hacia qué zona de una grilla
**3×3** miras y permite **seleccionar una frase** parpadeando. La app
corre en **pantalla completa**.

```text
[ tengo hambre ][ dejenme solo ][ tengo sed ]
[ si           ][ como estas   ][ no        ]
[ arriba       ][ me duele     ][ abajo     ]
```

No es un eye tracker profesional: el objetivo es distinguir 9 zonas
amplias y usar el parpadeo como “clic”, no precisión milimétrica.

---

## Inicio rápido (después de clonar desde GitHub)

### Requisitos

* **Python 3.12** (recomendado; MediaPipe no soporta bien 3.13/3.14)
* Webcam integrada o USB
* Internet **solo la primera vez** (descarga el modelo MediaPipe ~3.7 MB)
* En macOS: permiso de cámara para la app desde la que ejecutes
  (Terminal, Cursor, etc.)

### 1. Clonar el repositorio

```bash
git clone https://github.com/Juan-GuevaraBusta/eyetracker-communication.git
cd eyetracker-communication
```

### 2. Crear el entorno e instalar dependencias

**Opción recomendada (con [`uv`](https://github.com/astral-sh/uv)):**

```bash
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**Opción alternativa (venv estándar):**

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si no tienes Python 3.12:

* [python.org](https://www.python.org/downloads/), o
* `uv python install 3.12` / `pyenv install 3.12`

### 3. Ejecutar

```bash
python main.py
```

o, para ver mensajes de arranque al instante:

```bash
python -u main.py
```

* La **primera vez** descarga `face_landmarker.task` en `models/`.
* La ventana abre en **pantalla completa**.
* En macOS, acepta el permiso de cámara si el sistema lo pide.

### 4. Uso

1. Pantalla de instrucciones: coloca el rostro hasta ver **Rostro OK**
   (miniatura arriba a la derecha).
2. **ESPACIO** → calibración de 9 puntos (mira cada círculo solo con
   los ojos; la cabeza lo más quieta posible).
3. Tras la validación, entra a la grilla con las frases.
4. El **punto** sigue tu mirada. **Parpadea** (ambos ojos) para marcar
   de azul la casilla que estabas mirando.
5. Controles:

```text
ESC      salir
R        recalibrar
ESPACIO  iniciar calibración / confirmar mensajes
```

---

## Estructura del proyecto

```text
eyetracker-communication/
├── main.py                 # fullscreen, estados, UI, frases, selección por parpadeo
├── gaze.py                 # cámara + MediaPipe FaceLandmarker + mirada / blink
├── calibration.py          # 9 puntos, regresión lineal, validación
├── requirements.txt
├── test_mvp_calibration.py # pruebas unitarias básicas
├── models/                 # se crea sola; face_landmarker.task (no va al git)
└── README.md
```

Dependencias (`requirements.txt`):

* `mediapipe==0.10.21`
* `numpy>=1.26,<3`
* `screeninfo>=0.8.1`  
  (OpenCV llega como dependencia de MediaPipe)

---

## Tecnologías

* Python 3.12
* OpenCV (ventana fullscreen + cámara)
* **MediaPipe 0.10.21** — API `mediapipe.tasks.python.vision.FaceLandmarker`
* NumPy
* `screeninfo` (resolución real del monitor)

> La API antigua `mp.solutions.face_mesh` fue eliminada en MediaPipe
> >= 0.10.31. Este proyecto usa la API Tasks. Se fija `0.10.21` porque
> `1.0.1` en macOS/Apple Silicon puede fallar con un crash de GPU
> (`Service is unavailable`) incluso forzando CPU.

---

## Cómo funciona (resumen)

1. MediaPipe detecta el rostro y 478 landmarks (incluye iris).
2. Por cada ojo se calcula la posición relativa del iris dentro del
   contorno del ojo (`u`, `v`), más yaw/pitch aproximados de la cabeza.
3. En **calibración**, miras 9 puntos conocidos y se ajusta una
   regresión lineal (ojos → pantalla).
4. En **seguimiento**, se estima la casilla 1–9. Al **parpadear**, se
   rellena de azul la última casilla con ojos abiertos (donde iba el
   punto) y se muestra un marco negro breve alrededor de la grilla.

Detalles útiles:

* Parpadeo: ambos ojos con score de blendshape `> 0.25`
  (`eyeBlinkLeft` / `eyeBlinkRight`).
* El azul no usa el iris del frame con ojos cerrados (ese dato engaña);
  usa la última casilla estable con ojos abiertos.
* Las frases se dibujan centradas y a la escala máxima que cabe en cada
  celda.

---

## Comprobar la cámara (macOS)

Si `main.py` no abre, prueba solo la cámara:

```bash
python -c "import cv2; print('1'); c=cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION); print('2', c.isOpened()); print('3', c.read()[0]); c.release(); print('ok')"
```

Debes ver `1`, `2 True`, `3 True`, `ok`. Si eso funciona, el permiso
de cámara está bien.

---

## Problemas comunes

| Problema | Causa probable | Solución |
|---|---|---|
| `RuntimeError: No se pudo abrir la cámara` | Cámara en uso o sin permiso | Cierra Zoom/Meet/FaceTime; en macOS: Ajustes → Privacidad → Cámara (activa Cursor o Terminal) |
| Se queda “clavado” al arrancar | Antes: fullscreen antes de abrir la cámara | Actualiza el código (cámara / MediaPipe primero, luego fullscreen). Usa `python -u main.py` |
| No aparece Terminal en permisos de cámara | Ejecutas desde Cursor/IDE | El permiso que cuenta es el de **Cursor** (o la app que lanza Python), no Terminal.app |
| `ModuleNotFoundError: mediapipe` | venv no activado | `source .venv/bin/activate` y `pip install -r requirements.txt` |
| Falla instalar MediaPipe en Python 3.13/3.14 | Sin ruedas oficiales | Usa Python **3.12** |
| Crash `Service is unavailable` / Metal | `mediapipe` 1.x en Apple Silicon | `pip install mediapipe==0.10.21` |
| El azul marca otra casilla | Miraste otro sitio o la calibración falló | Espera a que el punto esté en la casilla deseada y parpadea; recalibra con `R` |
| Primera vez sin internet | Falta descargar el modelo | Conéctate una vez; luego `models/face_landmarker.task` queda local |

---

## Limitaciones

* Distingue **9 zonas**, no mirada milimétrica.
* Sensible a movimiento de cabeza, iluminación y distancia a la pantalla.
* Hay que **recalibrar** si cambias de postura o de cámara.
* Un solo rostro (el principal que detecte MediaPipe).

---

## Licencia / uso académico

Proyecto académico (PI2). Úsalo y adáptalo con fines educativos.
