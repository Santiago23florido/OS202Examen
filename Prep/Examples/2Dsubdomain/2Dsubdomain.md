# 2D Subdomain con MPI en Python

Este ejemplo vive en [2Dsubdomain.py](./2Dsubdomain.py) y muestra una descomposicion 2D del dominio para `Game of Life`.

La idea es la misma que en `RowsSubdomain`, pero ahora la grilla no se parte solo por filas. Se parte en bloques 2D.

## Que cambia respecto a rows subdomain

En `rows subdomain` cada proceso solo tenia:

- vecino de arriba,
- vecino de abajo.

En `2D subdomain` cada proceso puede tener vecindad en ambos ejes:

- arriba,
- abajo,
- izquierda,
- derecha,
- arriba-izquierda,
- arriba-derecha,
- abajo-izquierda,
- abajo-derecha.

Por eso aqui no alcanza con ghost rows. Tambien hacen falta:

- ghost columns,
- ghost corners.

## Idea general

Cada rank recibe un bloque rectangular de la grilla global.

Antes de cada iteracion:

1. envia y recibe la frontera superior e inferior,
2. envia y recibe la frontera izquierda y derecha,
3. envia y recibe las cuatro esquinas,
4. ejecuta el mismo kernel local de `Game of Life`,
5. repite.

## La idea de optimizacion que conviene conservar

La estrategia buena aqui es:

- dividir la grilla una sola vez,
- mantener el bloque local en cada proceso,
- intercambiar solo halos,
- no mover la grilla completa en cada paso,
- usar una topologia cartesiana periodica para que la vecindad sea natural.

## Estructura del codigo

Las piezas importantes son:

- `choose_process_grid(...)`
- `build_cart_comm(...)`
- `grid_distribution(...)`
- `get_neighbors(...)`
- `exchange_halos(...)`
- `Grille.compute_next_iteration()`
- `gather_global_grid(...)`
- `simulate_2d_subdomain(...)`

## Topologia cartesiana

El comunicador se crea como una grilla 2D de procesos:

```python
cart_comm = comm.Create_cart(dims=dims, periods=[True, True], reorder=False)
```

Eso significa:

- los procesos se organizan como matriz,
- la topologia es periodica en filas y columnas,
- el borde izquierdo conecta con el derecho,
- el borde superior conecta con el inferior.

Eso hace que el comportamiento sea equivalente al `boundary="wrap"` del problema global.

## Como se reparte la grilla

La matriz global se parte en dos dimensiones:

- unas filas para la coordenada de proceso `row`,
- unas columnas para la coordenada de proceso `col`.

Cada rank tiene:

- `local_rows`,
- `local_cols`,
- `offset_row`,
- `offset_col`.

Con eso puede construir su bloque local sin necesitar la grilla completa.

## Ghost cells

El bloque local queda asi:

```text
[ ghost corner ][ ghost top row    ][ ghost corner ]
[ ghost left   ][ core local block ][ ghost right  ]
[ ghost corner ][ ghost bottom row ][ ghost corner ]
```

El `core` es donde vive el subdominio real.
Las ghost cells solo sirven para poder calcular vecinos de borde.

## Intercambio de halos

La funcion importante es [exchange_halos(...) ](./2Dsubdomain.py).

Hace ocho intercambios logicos:

- fila superior,
- fila inferior,
- columna izquierda,
- columna derecha,
- esquina superior izquierda,
- esquina superior derecha,
- esquina inferior izquierda,
- esquina inferior derecha.

Eso permite que el kernel local vea correctamente los 8 vecinos de cada celda, incluso en bordes del subdominio.

## Kernel local

La logica del `Game of Life` se mantiene separada dentro de `Grille.compute_next_iteration()`.

La diferencia es que:

- para el caso local 2D se usa `convolve2d(..., boundary="fill")`,
- porque la periodicidad global ya no la resuelve `convolve2d`,
- la resuelven los halos intercambiados entre procesos.

## Plantilla conceptual reutilizable

Esta estructura sirve para otros problemas 2D:

```python
for step in range(steps):
    exchange_halos(cart_comm, local_block)
    local_block = compute_local_step(local_block)
```

Solo cambia `compute_local_step(...)`.

La infraestructura 2D casi no cambia.

## Cuando usar 2D subdomain

Conviene cuando:

- el problema realmente es 2D,
- cada celda depende de vecinos en filas y columnas,
- quieres escalar mejor que con una descomposicion solo por filas,
- el costo de comunicar filas y columnas compensa el mejor balance espacial.

## Errores tipicos

Los errores mas comunes son:

- intercambiar solo filas y olvidar columnas,
- olvidar las esquinas,
- usar `boundary="wrap"` dentro del subdominio despues de haber hecho halos,
- reconstruir mal los offsets globales,
- elegir una grilla de procesos incompatible con las dimensiones del dominio.

## Como ejecutar el ejemplo

Instala dependencias:

```bash
python3 -m pip install mpi4py numpy scipy
```

Ejecuta:

```bash
mpiexec -n 4 python3 Prep/Examples/2Dsubdomain/2Dsubdomain.py glider 200
```

Formato:

```bash
mpiexec -n <procesos> python3 Prep/Examples/2Dsubdomain/2Dsubdomain.py <pattern> <iteraciones>
```

## Resumen practico

Si quieres replicarlo rapido en otro problema 2D:

1. reparte el dominio en bloques,
2. crea una topologia cartesiana,
3. agrega ghost rows, ghost columns y esquinas,
4. intercambia halos en las 8 direcciones necesarias,
5. ejecuta el kernel local,
6. junta el resultado final solo si hace falta.

Ese es el patron central de `2D subdomain`.
