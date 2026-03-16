# Test Codes

Este directorio contiene cuatro scripts de prueba:

- `mandelbrot.py`: genera y muestra el conjunto de Mandelbrot con NumPy.
- `mandelbrot_numba.py`: misma idea, pero acelerada con Numba.
- `visualizer3d_sans_vbo.py`: demo de visualizacion 3D sin VBO.
- `visualizer3d_vbo.py`: demo de visualizacion 3D usando VBO.

## Requisitos

Se recomienda usar Python 3.12 o compatible.

Dependencias de Python:

```bash
python3 -m pip install numpy matplotlib numba PySDL2 PyOpenGL PyOpenGL_accelerate
```

En Debian/Ubuntu, los visualizadores 3D tambien suelen necesitar librerias del sistema:

```bash
sudo apt install libsdl2-2.0-0 libsdl2-dev libgl1-mesa-dev libglu1-mesa-dev
```

## Ejecucion

Desde la raiz del repo:

```bash
cd /home/santiago/OS202/Exam/Prep/test_codes
```

### 1. Mandelbrot con NumPy

```bash
python3 mandelbrot.py
```

Resultado:

- imprime el tiempo de ejecucion en consola
- guarda una imagen en `plot.png`
- abre una ventana con la figura

### 2. Mandelbrot con Numba

```bash
python3 mandelbrot_numba.py
```

Resultado:

- imprime el tiempo de ejecucion en consola
- guarda una imagen en `plot.png`
- abre una ventana con la figura

Nota:

- la primera ejecucion puede tardar mas porque Numba compila la funcion

### 3. Visualizador 3D sin VBO

```bash
python3 visualizer3d_sans_vbo.py
```

Controles:

- clic izquierdo + mover raton: rotar camara
- rueda del raton: zoom
- `Esc`: salir

### 4. Visualizador 3D con VBO

```bash
python3 visualizer3d_vbo.py
```

Controles:

- clic izquierdo + mover raton: rotar camara
- rueda del raton: zoom
- `Esc`: salir

## Problemas comunes

- Si `matplotlib` no abre ventana, prueba con un entorno grafico local o revisa la configuracion de display.
- Si `visualizer3d_vbo.py` falla por compatibilidad OpenGL/GPU, usa `visualizer3d_sans_vbo.py`.
- Si aparece un error de `sdl2` u OpenGL, revisa que las librerias del sistema esten instaladas.
