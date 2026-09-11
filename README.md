# Eye Tracker 3x3 (proyecto académico)

Eye tracker sencillo que usa la cámara web del computador para detectar
cuál de los 9 cuadros de una cuadrícula 3x3 está mirando el usuario. La
aplicación se ejecuta siempre en **pantalla completa real** del monitor.

```text
[ 1 ][ 2 ][ 3 ]
[ 4 ][ 5 ][ 6 ]
[ 7 ][ 8 ][ 9 ]
```

No es un eye tracker profesional: el objetivo es detectar la zona
aproximada de la pantalla que se está mirando, no una precisión
milimétrica.

---

## 1. Estructura del proyecto

```text
eyetracker/
├── main.py            # ventana fullscreen, máquina de estados, UI, controles
├── gaze.py            # cámara + MediaPipe FaceLandmarker + característica de mirada
├── calibration.py      # recolección de 9 puntos, regresión lineal, validación
├── requirements.txt
├── models/            # se crea sola; aquí se descarga face_landmarker.task
└── README.md
```

## 1.1 Mejoras de calibración basadas en documentación oficial

La calidad de la calibración se mejoró aplicando varias capacidades
**documentadas oficialmente** por Google para `FaceLandmarker`
([guía "Face landmark detection"](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)),
que antes no se usaban:

1. **Contorno completo del ojo** (16 puntos, índices oficiales
   `FACEMESH_RIGHT_EYE`/`FACEMESH_LEFT_EYE`) en vez de solo 2 esquinas,
   para un cálculo de ancho/alto del ojo más robusto al ruido.
2. **Rechazo de parpadeos** usando las categorías oficiales de
   blendshapes (compatibles con ARKit) `eyeBlinkLeft`/`eyeBlinkRight`
   (`output_face_blendshapes=True`), para no calibrar ni predecir con
   frames donde el ojo está cerrado.
3. **Compensación de pose de cabeza** (yaw/pitch) usando la matriz de
   transformación facial (`output_facial_transformation_matrixes=True`),
   agregada como características adicionales al vector de mirada.
4. **Agregación robusta** de las muestras de cada punto de calibración
   con mediana + recorte de outliers por MAD, y extensión automática
   del tiempo de captura si se descartan demasiadas muestras por
   parpadeo.

El detalle técnico completo de cada mejora está en la sección
["7. Cómo funciona internamente"](#7-cómo-funciona-internamente).

## 2. Tecnologías y versiones

* Python 3.12 (recomendado; ver nota de compatibilidad más abajo)
* OpenCV (vía `opencv-contrib-python`, dependencia de `mediapipe`)
* **MediaPipe 0.10.21** — usando la API vigente `mediapipe.tasks.python.vision.FaceLandmarker`
* NumPy
* `screeninfo` (para detectar la resolución real del monitor)

> **Nota importante sobre MediaPipe:** la API antigua `mp.solutions.face_mesh`
> fue **eliminada** en versiones recientes de MediaPipe (>= 0.10.31). Por eso
> este proyecto usa la API "Tasks" (`FaceLandmarker`), que es la vigente y
> recomendada oficialmente por Google. Requiere un archivo de modelo
> (`face_landmarker.task`, ~3.7 MB) que **se descarga automáticamente la
> primera vez que ejecutas `main.py`** (necesitas conexión a internet solo
> esa primera vez).
>
> Se fija la versión `mediapipe==0.10.21` (en lugar de la más reciente,
> 1.0.1) porque, al probar el proyecto, la versión 1.0.1 presenta un bug
> conocido en macOS/Apple Silicon: la subgráfica de detección de rostro
> falla con un error fatal de GPU (`Check failed: service_ Service is
> unavailable`) incluso forzando el delegado a CPU. La versión 0.10.21
> usa la misma API de Tasks y funciona correctamente.

> **Nota de compatibilidad de Python:** MediaPipe solo publica soporte
> oficial para Python 3.9-3.12. Si tu `python3` por defecto es 3.13 o 3.14,
> crea el entorno virtual explícitamente con Python 3.12 (ver instalación).

---

## 3. Instalación

### Opción recomendada (con `uv`)

```bash
uv venv --python 3.12
source .venv/bin/activate       # en Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Opción alternativa (con `venv` estándar)

macOS / Linux:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Si no tienes Python 3.12 instalado, instálalo desde
[python.org](https://www.python.org/downloads/) o con `pyenv`/`uv python install 3.12`.

---

## 4. Cómo ejecutar

```bash
python main.py
```

* La primera vez descargará el modelo `face_landmarker.task` en `models/`.
* La ventana se abrirá **en pantalla completa** automáticamente.
* En macOS, la primera vez que accedas a la cámara el sistema pedirá
  permiso para que la Terminal (o tu IDE) use la cámara: acéptalo y
  vuelve a ejecutar el programa si es necesario.

---

## 5. Proceso de calibración

1. Al iniciar, verás la pantalla de instrucciones. El sistema **no te
   deja avanzar** hasta detectar tu rostro (recuadro verde en la
   miniatura de la cámara, arriba a la derecha).
2. Presiona **ESPACIO** para comenzar la calibración de 9 puntos.
3. Para cada punto:
   * Aparece un punto rojo: dirige la mirada hacia él (no gires la
     cabeza, solo los ojos).
   * El punto se encoge durante ~1 segundo (tiempo para que ajustes la
     mirada) y luego cambia de color (naranja, parpadeante): en ese
     momento el sistema está capturando ~40 muestras durante ~1.5 s.
4. Al terminar los 9 puntos, el sistema valida el resultado mirando 5
   puntos adicionales (distintos a los de calibración) y calcula el
   error como **porcentaje de la diagonal de la pantalla**.
5. Si el error es aceptable (≤ 12%), pasa automáticamente al modo de
   seguimiento. Si es demasiado alto, te pide recalibrar (mejorando
   iluminación, distancia y estabilidad de la cabeza).

## 6. Controles

```text
ESC     -> salir del programa
R       -> recalibrar (vuelve a la pantalla de instrucciones)
ESPACIO -> iniciar calibración / confirmar mensajes
```

---

## 7. Cómo funciona internamente

### 7.1 Detección del rostro

`gaze.py` usa `mediapipe.tasks.python.vision.FaceLandmarker` en modo
`VIDEO` (procesa frame por frame, entregando 478 landmarks 3D por
rostro). Se configura `num_faces=1` para que, si hay varias personas
frente a la cámara, MediaPipe se quede solo con el rostro principal.

### 7.2 Detección de los ojos y del iris

De los 478 landmarks, se usan:

* El **contorno completo de cada ojo** (16 puntos por ojo), usando los
  índices oficiales `FACEMESH_RIGHT_EYE` / `FACEMESH_LEFT_EYE` del
  módulo `mediapipe.python.solutions.face_mesh_connections`. Usar los
  16 puntos (en vez de solo 2 esquinas) para calcular el ancho/alto del
  ojo es más robusto: un único landmark con ruido afecta mucho menos al
  resultado.
* 2 grupos de 5 puntos que forman el contorno del iris (habilitados
  automáticamente por el modelo de landmarks faciales).

Como la correspondencia "grupo de iris -> ojo izquierdo/derecho" no es
siempre obvia, el código la determina dinámicamente comparando la
posición X del centroide de cada grupo de iris con la de cada ojo
anatómico, y lo asigna al más cercano.

### 7.3 Obtención de la posición de la mirada (característica)

Para cada ojo se calcula una posición relativa del iris dentro de la
**caja delimitadora de todo el contorno del ojo**:

```text
u = (iris_x - min_x_contorno) / (max_x_contorno - min_x_contorno)
v = (iris_y - min_y_contorno) / (max_y_contorno - min_y_contorno)
```

A esto se agregan dos valores más, obtenidos de la **matriz de
transformación facial** que MediaPipe puede entregar activando
`output_facial_transformation_matrixes=True` (documentado en la guía
oficial de "Face landmark detection"): un ángulo aproximado de **yaw**
(giro izquierda/derecha de la cabeza) y de **pitch** (giro arriba/abajo),
extraídos de la submatriz de rotación 3x3.

El vector final que representa la mirada es:

```text
[u_derecho, v_derecho, u_izquierdo, v_izquierdo, yaw, pitch]
```

Se usan **razones dentro del propio ojo** (y no coordenadas absolutas
en la imagen) porque así, si la cabeza se mueve un poco, tanto el iris
como el contorno del ojo se desplazan de forma parecida en la imagen,
y el cociente `u`/`v` cambia mucho menos que la posición absoluta.
Agregar `yaw`/`pitch` como características adicionales le da a la
regresión de calibración una señal explícita del giro de cabeza, para
que pueda aprender a compensar su efecto sobre la razón iris/ojo (en
vez de depender solo de la normalización geométrica). Esto reduce (sin
resolver por completo) el efecto del movimiento de cabeza, como pide
la sección 12 del prompt original.

### 7.3.1 Rechazo de parpadeos (blendshapes oficiales)

Durante un parpadeo, la posición del iris dentro del ojo no significa
nada respecto a hacia dónde mira la persona, y usar esos frames
contamina tanto la calibración como el seguimiento. Para detectarlos
se activa `output_face_blendshapes=True` y se leen las categorías
estándar (compatibles con ARKit, documentadas por MediaPipe)
`eyeBlinkLeft` y `eyeBlinkRight`: si alguna supera 0.5, el frame se
marca como parpadeo y:

* durante la **calibración**, esa muestra no se usa para el promedio
  del punto;
* durante el **seguimiento**, se mantiene la última predicción estable
  en pantalla en vez de actualizarla con datos ruidosos.

Como filtrar parpadeos reduce el número de muestras válidas por punto
de calibración, la fase de captura se **extiende automáticamente**
(hasta ~2.5 s adicionales) si no se alcanzan al menos 15 muestras
válidas, en vez de calibrar ese punto con muy pocos datos.

### 7.3.2 Agregación robusta de las muestras de cada punto

Las muestras válidas de un punto de calibración se combinan con
mediana + recorte de outliers por MAD (*median absolute deviation*):
se descarta cualquier muestra que se aleje más de 3.5 MADs de la
mediana en alguna de sus 6 componentes, y luego se recalcula la
mediana solo con las muestras restantes. Esto es más robusto que una
mediana simple cuando quedan parpadeos aislados que el filtro de
blendshapes no detectó a tiempo (por ejemplo, en el frame justo antes
o después del pico de la categoría de blendshape).

### 7.4 Por qué es necesaria la calibración

Cada persona tiene una geometría facial distinta: tamaño de ojos,
distancia a la cámara, posición de la cámara respecto a la pantalla,
etc. Un mismo valor de `u`/`v` significa cosas distintas para cada
persona, por lo que **no se pueden usar umbrales fijos**. En cambio, se
le pide al usuario que mire 9 puntos conocidos: así el sistema aprende,
para esa persona y esa sesión concreta, la relación entre la
característica `[u_der, v_der, u_izq, v_izq]` y la posición real en la
pantalla.

### 7.5 De los 9 puntos a la transformación ojos -> pantalla

Se ajusta un modelo **afín** (regresión lineal + término independiente)
por mínimos cuadrados, con una pequeña regularización ridge para
estabilidad numérica:

```text
[screen_x, screen_y] = [u_der, v_der, u_izq, v_izq, yaw, pitch, 1] · theta
theta = (X^T X + lambda*I)^-1 X^T Y      (lambda = 1e-3)
```

Con 9 puntos de calibración y 7 parámetros por coordenada (6 pesos + 1
sesgo), el sistema sigue sobre-determinado (hay más puntos que
parámetros), lo que ayuda a promediar el ruido de las mediciones
individuales. Se evitó a propósito usar términos cuadráticos o
agregar muchas más características: con solo 9 puntos de calibración,
un modelo con demasiados parámetros tiende a sobreajustar (memoriza
los 9 puntos pero generaliza mal), que es precisamente lo que la
validación con 5 puntos nuevos está pensada para detectar.

### 7.6 De la coordenada estimada al cuadro 1-9

```python
col = clip(int(screen_x / (screen_w / 3)), 0, 2)
row = clip(int(screen_y / (screen_h / 3)), 0, 2)
cuadro = row * 3 + col + 1
```

### 7.7 Cómo se evita que la detección "parpadee"

Se combinan dos técnicas (sección 11 del prompt):

1. **Suavizado exponencial** del punto de mirada estimado:
   `punto = alpha * nuevo + (1 - alpha) * anterior` (alpha = 0.35),
   que reduce el ruido frame a frame antes de decidir el cuadro.
2. **Voto mayoritario** sobre las últimas 10 predicciones de cuadro
   (`collections.Counter` sobre un `deque` de tamaño 10): el cuadro que
   se muestra es el que más veces apareció en esa ventana reciente, no
   necesariamente el del frame actual.

---

## 8. Cómo comprobar que detecta correctamente los 9 cuadros

Una vez en modo de seguimiento:

1. Mira fijamente cada una de las 9 zonas de la pantalla (por ejemplo,
   siguiendo el mismo patrón 20/50/80% usado en la calibración) y
   verifica en el HUD (`Cuadro: N`) que el número coincide con la zona
   que estás mirando.
2. Observa el punto azul dibujado en pantalla: debe moverse de forma
   coherente con tu mirada.
3. Revisa el FPS reportado: en un computador normal debería mantenerse
   alrededor de 20-30 FPS o más.
4. Si notas que un cuadro se detecta sistemáticamente desplazado (por
   ejemplo, siempre marca el cuadro de la derecha cuando miras el
   centro), recalibra con la tecla `R`, procurando no mover la cabeza
   entre la calibración y el uso.

---

## 9. Problemas comunes

| Problema | Causa probable | Solución |
|---|---|---|
| `RuntimeError: No se pudo abrir la cámara web` | Cámara en uso por otra app, o sin permisos | Cierra otras apps que usen la cámara; en macOS revisa Ajustes del Sistema > Privacidad y Seguridad > Cámara |
| El programa se queda descargando el modelo | Sin conexión a internet en el primer arranque | Conéctate a internet solo la primera vez; luego el modelo queda en `models/` |
| `ModuleNotFoundError: No module named 'mediapipe'` | El entorno virtual no está activado o mediapipe no se instaló | Activa el entorno (`source venv/bin/activate`) y reinstala con `pip install -r requirements.txt` |
| Falla la instalación de `mediapipe` en Python 3.13/3.14 | MediaPipe no publica ruedas oficiales para esas versiones | Usa Python 3.12 para crear el entorno virtual |
| La ventana no ocupa toda la pantalla | El gestor de ventanas del sistema restringe el fullscreen de OpenCV | Verifica que no haya otra ventana forzando el foco; en algunos Linux con Wayland puede requerir X11 |
| El cuadro detectado "salta" constantemente | Iluminación pobre, cabeza muy inclinada o demasiado cerca/lejos | Mejora la iluminación frontal del rostro y recalibra con `R` |
| Error de calibración siempre alto | El usuario movió la cabeza durante los 9 puntos, o miró de reojo | Recalibra manteniendo la cabeza quieta y moviendo solo los ojos |
| Crash fatal `Check failed: service_ Service is unavailable` / `DrishtiMetalHelper` | Estás usando `mediapipe==1.0.1` en macOS/Apple Silicon, versión con un bug conocido en la subgráfica de detección de rostro | Instala la versión fijada en `requirements.txt` (`pip install mediapipe==0.10.21`), que es la que usa este proyecto por defecto |

---

## 10. Limitaciones

* **No es un eye tracker de precisión profesional.** Está pensado para
  distinguir 9 zonas amplias de la pantalla, no puntos exactos.
* **Sensible al movimiento de cabeza.** Aunque se usan coordenadas
  relativas dentro del ojo para mitigar esto, mover mucho la cabeza
  después de calibrar (o cambiar de postura) degrada la precisión.
* **Sensible a la iluminación.** Poca luz o luz muy lateral dificulta
  que MediaPipe detecte los landmarks del iris con precisión.
* **Depende de la calibración de cada sesión.** Si cambias de silla,
  de distancia a la pantalla o de posición de la cámara, debes
  recalibrar (`R`).
* **Un solo rostro.** Si hay varias personas frente a la cámara, solo
  se sigue al rostro principal detectado por MediaPipe; no distingue
  "de quién" es la mirada si hay más de una persona.
* **El modelo lineal es intencionalmente simple.** Una transformación
  afín no captura relaciones no lineales complejas (por ejemplo,
  distorsión de lente en los bordes de la imagen de la cámara), pero es
  suficiente para distinguir 9 zonas amplias como pide este proyecto.
