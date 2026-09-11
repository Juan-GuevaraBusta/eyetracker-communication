Actúa como un ingeniero experto en Computer Vision y Python, pero desarrolla este proyecto con el nivel de complejidad mínimo necesario para un proyecto académico.

Quiero construir un **eye tracker sencillo usando la cámara web de un computador** cuyo objetivo sea detectar cuál de los 9 cuadros de una **matriz 3×3** está mirando el usuario.

## 1. Objetivo

La aplicación debe mostrarse SIEMPRE en pantalla completa (fullscreen) del computador, ocupando el 100% de la pantalla, y la cuadrícula 3×3 debe ocupar toda esa área disponible.

La cuadrícula debe estar delimitada por **líneas negras claramente visibles**.

El sistema debe utilizar la cámara web para estimar hacia qué zona de la pantalla está mirando el usuario y determinar uno de estos 9 resultados:

```text
[ 1 ][ 2 ][ 3 ]
[ 4 ][ 5 ][ 6 ]
[ 7 ][ 8 ][ 9 ]
```

Por ejemplo:

* esquina superior izquierda → cuadro 1
* centro superior → cuadro 2
* esquina superior derecha → cuadro 3
* centro izquierda → cuadro 4
* centro → cuadro 5
* centro derecha → cuadro 6
* esquina inferior izquierda → cuadro 7
* centro inferior → cuadro 8
* esquina inferior derecha → cuadro 9

El objetivo NO es crear un eye tracker profesional con precisión milimétrica. Solo necesito determinar correctamente la zona aproximada de la pantalla que está mirando el usuario.

---

# 2. Tecnología

Utiliza preferiblemente:

* Python
* OpenCV
* MediaPipe Face Mesh o una alternativa sencilla disponible actualmente
* NumPy

Evita utilizar frameworks innecesariamente complejos.

No utilices machine learning personalizado ni entrenes un modelo desde cero.

No necesito una aplicación web.

La cámara debe ser la webcam integrada del computador.

---

# 3. Arquitectura general

Implementa el sistema siguiendo este flujo:

```text
INICIO
   ↓
Abrir cámara
   ↓
Detectar rostro
   ↓
Detectar ojos / landmarks faciales
   ↓
CALIBRACIÓN
   ↓
Usuario mira 9 puntos
   ↓
Guardar datos de cada punto
   ↓
Crear transformación entre
posición de los ojos → posición en pantalla
   ↓
Modo de detección
   ↓
Estimar posición de mirada
   ↓
Determinar cuadro 1–9
   ↓
Resaltar cuadro detectado
```

Mantén todo el sistema en el menor número razonable de archivos.

Preferiblemente:

```text
eye_tracker/
│
├── main.py
├── calibration.py
├── gaze.py
└── requirements.txt
```

Si es posible hacerlo correctamente con menos archivos, hazlo.

---

# 4. Interfaz

La aplicación debe abrir la ventana en modo PANTALLA COMPLETA REAL desde el inicio, y mantenerse así durante la calibración, la validación y el modo de detección.

Requisitos de la pantalla completa:

* La ventana debe ocupar todo el monitor, sin bordes, barra de título ni márgenes.
* En OpenCV usa una ventana con `cv2.namedWindow("...", cv2.WINDOW_NORMAL)` y `cv2.setWindowProperty("...", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)`.
* Obtén la resolución real del monitor de forma automática (por ejemplo con `tkinter` o `screeninfo`) y usa ese ancho y alto para dibujar el lienzo de la cuadrícula, en lugar de valores fijos como 1280×720.
* El tamaño del frame de la cámara (640×480) es independiente del tamaño de la ventana: la cámara se procesa en baja resolución, pero la interfaz se dibuja a la resolución completa de la pantalla.
* Los 9 puntos de calibración (20%/50%/80%) deben calcularse sobre el ancho y alto reales de la pantalla en pantalla completa.
* ESC debe cerrar la ventana y salir del modo pantalla completa correctamente.

Debe mostrar una matriz 3×3 con líneas negras.

Ejemplo conceptual:

```text
┌───────────┬───────────┬───────────┐
│           │           │           │
│     1     │     2     │     3     │
│           │           │           │
├───────────┼───────────┼───────────┤
│           │           │           │
│     4     │     5     │     6     │
│           │           │           │
├───────────┼───────────┼───────────┤
│           │           │           │
│     7     │     8     │     9     │
│           │           │           │
└───────────┴───────────┴───────────┘
```

Las líneas de separación deben ser negras y suficientemente gruesas para que sean claramente visibles.

Cuando el sistema determine que el usuario está mirando un cuadro, ese cuadro debe resaltarse visualmente.

Por ejemplo:

```text
┌───────────┬───────────┬───────────┐
│           │           │           │
│     1     │     2     │     3     │
│           │           │           │
├───────────┼───────────┼───────────┤
│           │███████████│           │
│     4     │██████ 5 ██│     6     │
│           │███████████│           │
├───────────┼───────────┼───────────┤
│           │           │           │
│     7     │     8     │     9     │
│           │           │           │
└───────────┴───────────┴───────────┘
```

No necesitas mostrar el número si no es necesario, pero sí debe ser evidente qué cuadro está siendo detectado.

---

# 5. CALIBRACIÓN — PARTE FUNDAMENTAL

Antes de detectar la mirada, la aplicación debe realizar una calibración.

Esta etapa es obligatoria.

La razón es que cada persona tiene diferentes:

* posición frente a la cámara
* distancia a la pantalla
* tamaño de ojos
* inclinación de cabeza
* posición de la cámara
* características faciales

Por eso NO quiero utilizar valores fijos como:

```python
if eye_x < 100:
    ...
```

La aplicación debe aprender la relación entre la posición observada de los ojos y la posición conocida en pantalla.

---

# 6. Proceso de calibración

La calibración debe utilizar 9 puntos.

Los puntos deben estar aproximadamente en:

```text
(20%, 20%)    (50%, 20%)    (80%, 20%)

(20%, 50%)    (50%, 50%)    (80%, 50%)

(20%, 80%)    (50%, 80%)    (80%, 80%)
```

No coloques los puntos exactamente en los bordes de la pantalla.

Cada punto debe aparecer individualmente.

Ejemplo:

```text
Mira el punto rojo

          ●
```

El usuario debe mirar directamente el punto.

Después de unos segundos, el sistema debe guardar la información correspondiente.

Por ejemplo:

```text
Punto 1 → usuario mira arriba-izquierda
Punto 2 → usuario mira arriba-centro
Punto 3 → usuario mira arriba-derecha

...

Punto 9 → usuario mira abajo-derecha
```

---

# 7. Tiempo de calibración

Para cada punto:

1. Mostrar el punto.
2. Esperar aproximadamente 1 segundo para que el usuario dirija la mirada.
3. Capturar datos durante aproximadamente 1–2 segundos.
4. Calcular un promedio de las mediciones.
5. Guardar el promedio.
6. Pasar al siguiente punto.

No dependas de una única medición.

Es preferible obtener aproximadamente 20–50 muestras por punto y calcular el promedio.

---

# 8. Qué información guardar

Utiliza los landmarks de los ojos para obtener una representación sencilla de la mirada.

Por ejemplo, calcula:

* centro del ojo izquierdo
* centro del ojo derecho
* posición relativa de la pupila/iris si está disponible
* posición relativa respecto al ojo

La característica final puede ser algo sencillo como:

```text
gaze_x
gaze_y
```

o un pequeño vector:

```text
[
    eye_left_x,
    eye_left_y,
    eye_right_x,
    eye_right_y
]
```

Normaliza las coordenadas cuando sea necesario.

Quiero que expliques claramente qué característica estás utilizando y por qué.

---

# 9. Relación entre mirada y pantalla

Después de obtener los 9 puntos de calibración, crea una transformación que permita convertir:

```text
posición observada de los ojos
             ↓
posición estimada en pantalla
```

Para mantener el proyecto sencillo, utiliza preferiblemente una de estas opciones:

### Opción preferida

Regresión lineal:

```text
screen_x = f(gaze_features)
screen_y = f(gaze_features)
```

o una transformación afín si resulta más apropiada.

### Alternativa

Homografía si las características utilizadas representan correctamente coordenadas espaciales.

No implementes algoritmos complejos si una regresión lineal o transformación afín funciona.

---

# 10. Determinar el cuadro

Una vez obtenida la coordenada estimada:

```text
screen_x
screen_y
```

divide la pantalla en 3 columnas y 3 filas.

Por ejemplo:

```python
column = int(screen_x / (screen_width / 3))
row = int(screen_y / (screen_height / 3))
```

Limita los valores para que:

```text
column ∈ {0,1,2}
row ∈ {0,1,2}
```

Después convierte:

```text
row 0, column 0 → cuadro 1
row 0, column 1 → cuadro 2
row 0, column 2 → cuadro 3

row 1, column 0 → cuadro 4
...
```

---

# 11. Estabilidad

NO quiero que el sistema cambie constantemente de cuadro debido a pequeñas variaciones.

Por ejemplo:

```text
5
5
5
5
6
5
5
```

no debería hacer que la interfaz parpadee entre 5 y 6.

Implementa una estabilización sencilla.

Puedes utilizar:

* promedio de las últimas N predicciones
* mayoría de las últimas N predicciones
* o exigir que el mismo cuadro aparezca durante cierto número de frames

Por ejemplo:

```text
últimas 10 predicciones:

5 5 5 5 5 5 6 5 5 5

resultado → 5
```

Utiliza la solución más sencilla.

---

# 12. Movimiento de cabeza

El usuario no debe tener que mantener la cabeza perfectamente inmóvil.

El sistema debe intentar utilizar coordenadas relativas de los ojos/iris para que pequeños movimientos de cabeza no destruyan completamente la estimación.

Sin embargo, NO necesito resolver completamente el head pose estimation.

El objetivo es que el sistema funcione razonablemente bien mientras:

* el usuario permanece frente a la cámara
* no mueve excesivamente la cabeza
* mantiene una distancia aproximadamente constante de la pantalla

---

# 13. Condiciones previas

Antes de iniciar la calibración, muestra instrucciones:

```text
CALIBRACIÓN

1. Siéntate frente a la cámara.
2. Mantén la cabeza relativamente quieta.
3. Mantén una distancia constante de la pantalla.
4. Asegúrate de que tu rostro esté bien iluminado.
5. No muevas la cámara después de calibrar.
6. Sigue los puntos únicamente con los ojos.
7. Mira directamente cada punto cuando aparezca.

Presiona ESPACIO para comenzar.
```

Si no se detecta el rostro:

```text
No se detecta el rostro.
Ajusta tu posición frente a la cámara.
```

No permitas iniciar la calibración hasta detectar correctamente el rostro.

---

# 14. Validación de calibración

Después de calibrar, realiza una pequeña prueba.

Muestra nuevamente algunos puntos conocidos y compara:

```text
posición real del punto
vs.
posición estimada por el eye tracker
```

Calcula un error aproximado en píxeles o porcentaje.

Por ejemplo:

```text
Error promedio: 8.7% de la pantalla
```

Si el error es demasiado alto, mostrar:

```text
La calibración no fue suficientemente precisa.

Intenta nuevamente:
- mantener la cabeza quieta
- mirar directamente cada punto
- mejorar la iluminación
- mantener la cámara estable
```

Permite recalibrar presionando una tecla.

Define un umbral razonable y explícitalo en el código.

---

# 15. Visualización durante el funcionamiento

Durante el modo normal quiero poder ver:

* la matriz 3×3
* el cuadro actualmente detectado
* opcionalmente un pequeño punto que represente la posición estimada de la mirada
* FPS
* estado del sistema

Ejemplo:

```text
Gaze: (812, 421)
Cuadro: 5
FPS: 29
```

Esto es importante para poder demostrar el proyecto y comprobar visualmente que funciona.

---

# 16. Controles

Implementa controles sencillos:

```text
ESC → salir
R   → recalibrar
```

Si es útil:

```text
SPACE → iniciar calibración
```

---

# 17. Rendimiento

La aplicación debe funcionar en tiempo real con una webcam normal.

No proceses imágenes innecesariamente grandes.

Utiliza aproximadamente:

```text
640 × 480
```

si es suficiente.

El objetivo es conseguir aproximadamente 20–30 FPS o más en un computador normal.

---

# 18. Robustez

Maneja correctamente:

* cámara no disponible
* rostro no detectado
* múltiples rostros
* pérdida temporal del rostro
* landmarks incompletos
* errores de MediaPipe/OpenCV
* cierre de la ventana

Si aparecen múltiples rostros, utiliza únicamente el rostro principal/más grande.

---

# 19. Código

Quiero código REAL y ejecutable.

No quiero pseudocódigo.

Cada archivo debe estar completo.

Incluye:

```text
requirements.txt
```

con las dependencias necesarias.

También proporciona exactamente cómo instalar y ejecutar:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Si el sistema operativo requiere comandos diferentes, indícalos.

---

# 20. Compatibilidad

El código debe estar preparado para funcionar actualmente.

Antes de elegir APIs de MediaPipe/OpenCV, verifica que las APIs utilizadas sigan siendo válidas.

Evita utilizar funciones obsoletas.

Si existe una diferencia importante entre versiones de MediaPipe, utiliza la API actual y explícame qué versión estás utilizando.

---

# 21. Explicación académica

Después del código, explica de manera sencilla:

1. Cómo se detecta el rostro.
2. Cómo se detectan los ojos.
3. Cómo se obtiene la posición de la mirada.
4. Por qué es necesaria la calibración.
5. Cómo los 9 puntos permiten relacionar ojos y pantalla.
6. Cómo se transforma la coordenada estimada en uno de los 9 cuadros.
7. Cómo se evita que la detección sea inestable.
8. Cuáles son las limitaciones del sistema.

Utiliza ejemplos sencillos.

---

# 22. IMPORTANTE: simplicidad

Este es un proyecto académico y NO un producto comercial.

Por lo tanto:

* no agregues funcionalidades innecesarias
* no entrenes modelos
* no uses redes neuronales personalizadas
* no implementes servidores
* no uses bases de datos
* no uses una arquitectura excesivamente compleja
* no crees una interfaz innecesariamente sofisticada
* no agregues login
* no uses Docker
* no uses APIs externas

Quiero que el proyecto sea fácil de:

1. entender
2. ejecutar
3. presentar
4. modificar
5. depurar

Prioriza **simplicidad + funcionamiento + demostrabilidad**.

---

# 23. Orden de tu respuesta

Entrega la solución exactamente en este orden:

### 1. Concepto general

Explica brevemente cómo funcionará.

### 2. Tecnologías

Lista las tecnologías y versiones utilizadas.

### 3. Estructura del proyecto

Muestra el árbol de archivos.

### 4. Instalación

Explica paso a paso cómo instalar las dependencias.

### 5. Código completo

Entrega todos los archivos completos.

### 6. Cómo ejecutar

Explica exactamente cómo iniciar el programa.

### 7. Proceso de calibración

Explica exactamente qué debe hacer el usuario.

### 8. Cómo funciona internamente

Explica el algoritmo de manera sencilla.

### 9. Prueba

Explica cómo comprobar que realmente detecta correctamente los 9 cuadros.

### 10. Problemas comunes

Incluye errores frecuentes y soluciones.

### 11. Limitaciones

Explica honestamente qué tan preciso puede ser este sistema y qué factores afectan su funcionamiento.

No omitas la calibración. Es la parte más importante del proyecto.
