# Rows Subdomain con MPI en Python

Este documento acompana el ejemplo de [RowsSubdomain.py](./RowsSubdomain.py) y explica el patron de descomposicion por filas para problemas sobre mallas 2D.

En esta version se conserva la logica de `Game of Life` dentro de `Grille.compute_next_iteration()`. Lo que se corrige es solo el tratamiento de subdominios: reparto por filas, ghost rows, intercambio entre vecinos y recoleccion final.

## Idea general

`rows subdomain` significa partir la malla global por bloques de filas.

Cada proceso:

- recibe un subconjunto contiguo de filas,
- calcula solo sobre ese bloque,
- intercambia filas frontera con sus vecinos,
- repite el proceso en cada iteracion.

Esta estrategia sirve para algoritmos tipo stencil, por ejemplo:

- Game of Life,
- difusion de calor,
- Jacobi,
- relajacion,
- convoluciones 2D iterativas.

## La idea de optimizacion que conviene conservar

La optimizacion buena de este patron es:

- dividir la grilla una sola vez,
- mantener los datos localmente durante muchas iteraciones,
- intercambiar solo las filas frontera,
- evitar mover la grilla completa en cada paso.

Eso reduce mucho la comunicacion respecto a reenviar toda la matriz cada vez.

## Estructura del algoritmo

1. Se calcula cuantas filas le corresponden a cada rank.
2. Cada rank construye su subgrilla local a partir del pattern global y de su `offset` de filas.
3. Cada rank reserva ghost rows arriba y abajo.
4. En cada iteracion:
   - intercambia ghost rows con vecino superior e inferior,
   - ejecuta el mismo `compute_next_iteration()` de `Game of Life`.
5. `Gatherv` reconstruye la grilla final.

## Concepto clave: ghost rows

Si tu bloque local tiene estas filas:

```text
[ghost_top]
[fila local 0]
[fila local 1]
...
[fila local n-1]
[ghost_bottom]
```

entonces:

- `ghost_top` guarda la ultima fila util del rank anterior,
- `ghost_bottom` guarda la primera fila util del rank siguiente.

Con eso puedes calcular celdas de borde local sin necesitar la malla completa.

## Plantilla general copiable

```python
from mpi4py import MPI
import numpy as np

ROOT = 0
GHOSTS = 1


def static_row_distribution(total_rows, size, rank):
    base = total_rows // size
    rem = total_rows % size
    return base + 1 if rank < rem else base


def build_counts_displs(total_rows, size):
    counts = np.array(
        [static_row_distribution(total_rows, size, rank) for rank in range(size)],
        dtype=np.intc,
    )
    displs = np.zeros(size, dtype=np.intc)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


def scatter_rows(comm, rank, size, global_grid, total_cols):
    if rank == ROOT:
        total_rows = global_grid.shape[0]
    else:
        total_rows = None

    total_rows = comm.bcast(total_rows, root=ROOT)
    counts, displs = build_counts_displs(total_rows, size)

    sendcounts = (counts * total_cols).astype(np.intc)
    senddispls = (displs * total_cols).astype(np.intc)
    local_rows = counts[rank]

    local_core = np.empty((local_rows, total_cols), dtype=np.uint8)
    comm.Scatterv(
        [global_grid, sendcounts, senddispls, MPI.UNSIGNED_CHAR],
        [local_core, MPI.UNSIGNED_CHAR],
        root=ROOT,
    )

    local_grid = np.zeros((local_rows + 2 * GHOSTS, total_cols), dtype=np.uint8)
    local_grid[GHOSTS:-GHOSTS, :] = local_core
    return local_grid


def exchange_ghost_rows(comm, rank, size, local_grid):
    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size

    comm.Sendrecv(
        sendbuf=[local_grid[1, :], MPI.UNSIGNED_CHAR],
        dest=prev_rank,
        sendtag=10,
        recvbuf=[local_grid[-1, :], MPI.UNSIGNED_CHAR],
        source=next_rank,
        recvtag=10,
    )
    comm.Sendrecv(
        sendbuf=[local_grid[-2, :], MPI.UNSIGNED_CHAR],
        dest=next_rank,
        sendtag=11,
        recvbuf=[local_grid[0, :], MPI.UNSIGNED_CHAR],
        source=prev_rank,
        recvtag=11,
    )


def compute_next_iteration(local_grid):
    core = local_grid[1:-1, :]
    north = local_grid[:-2, :]
    south = local_grid[2:, :]

    neighbors = (
        north
        + south
        + np.roll(core, 1, axis=1)
        + np.roll(core, -1, axis=1)
        + np.roll(north, 1, axis=1)
        + np.roll(north, -1, axis=1)
        + np.roll(south, 1, axis=1)
        + np.roll(south, -1, axis=1)
    )

    next_core = (
        ((core == 1) & ((neighbors == 2) | (neighbors == 3)))
        | ((core == 0) & (neighbors == 3))
    ).astype(np.uint8)

    local_grid[1:-1, :] = next_core


def gather_rows(comm, rank, size, local_grid, total_cols):
    local_core = local_grid[1:-1, :]
    local_rows = local_core.shape[0]
    counts = comm.gather(local_rows, root=ROOT)

    if rank == ROOT:
        counts = np.array(counts, dtype=np.intc)
        displs = np.zeros(size, dtype=np.intc)
        displs[1:] = np.cumsum(counts[:-1])
        global_grid = np.empty((np.sum(counts), total_cols), dtype=np.uint8)
        recvcounts = (counts * total_cols).astype(np.intc)
        recvdispls = (displs * total_cols).astype(np.intc)
    else:
        global_grid = None
        recvcounts = None
        recvdispls = None

    comm.Gatherv(
        [local_core, MPI.UNSIGNED_CHAR],
        [global_grid, recvcounts, recvdispls, MPI.UNSIGNED_CHAR],
        root=ROOT,
    )
    return global_grid
```

## Ejemplo concreto: Game of Life

En el ejemplo del repo:

- la malla se reparte por filas,
- el kernel de `Game of Life` sigue dentro de `Grille.compute_next_iteration()`,
- el conteo de vecinos sigue usando `convolve2d(..., boundary="wrap")`,
- el borde vertical entre procesos se resuelve con ghost rows,
- cada iteracion hace `exchange_ghost_rows(...)` y despues `grille.compute_next_iteration()`.

Ese es el patron que debes conservar.

## Funcion de intercambio

La pieza mas importante es esta:

```python
def exchange_ghost_rows(comm, rank, size, local_grid):
    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size

    comm.Sendrecv(
        sendbuf=[local_grid[1, :], MPI.UNSIGNED_CHAR],
        dest=prev_rank,
        sendtag=10,
        recvbuf=[local_grid[-1, :], MPI.UNSIGNED_CHAR],
        source=next_rank,
        recvtag=10,
    )
    comm.Sendrecv(
        sendbuf=[local_grid[-2, :], MPI.UNSIGNED_CHAR],
        dest=next_rank,
        sendtag=11,
        recvbuf=[local_grid[0, :], MPI.UNSIGNED_CHAR],
        source=prev_rank,
        recvtag=11,
    )
```

Eso evita deadlocks y deja actualizadas las filas fantasma antes del calculo.

## Como adaptar este patron a otros algoritmos

Puedes reutilizarlo casi igual en cualquier stencil 2D.

Solo cambias:

- la funcion de actualizacion local,
- la cantidad de ghost rows si el stencil alcanza mas lejos,
- el tipo de dato,
- la condicion de borde si no quieres topologia toroidal.

Ejemplos:

- calor 2D: actualizas con promedio de vecinos,
- Jacobi: calculas nueva malla sin sobrescribir la anterior,
- difusion: aplicas el operador discreto sobre cada bloque local,
- filtros iterativos: usas la misma idea de halos.

## Errores tipicos en rows subdomain

Estos son los fallos mas comunes:

- repartir filas con `counts` mal calculados,
- olvidar multiplicar por el numero de columnas en `Scatterv` o `Gatherv`,
- calcular el siguiente estado antes de actualizar ghost rows,
- usar una sola comunicacion mal orientada y mezclar frontera superior e inferior,
- no reservar ghost rows y luego leer fuera del bloque local.

## Archivo corregido del repo

El ejemplo corregido mantiene funciones separadas para que el flujo se lea claro:

- `static_row_distribution(...)`
- `build_counts_displs(...)`
- `Grille`
- `grid_distribution(...)`
- `exchange_ghost_rows(...)`
- `gather_grid_rows(...)`
- `simulate_rows_subdomain(...)`
- `simulate_serial(...)`
- `main()`

Archivo: [RowsSubdomain.py](./RowsSubdomain.py)

## Como ejecutar el ejemplo

Instala dependencias:

```bash
python3 -m pip install mpi4py numpy scipy
```

Ejecuta:

```bash
mpiexec -n 4 python3 Prep/Examples/RowsSubdomain/RowsSubdomain.py glider 200
```

Formato:

```bash
mpiexec -n <procesos> python3 Prep/Examples/RowsSubdomain/RowsSubdomain.py <pattern> <iteraciones>
```

## Resumen practico

Si quieres replicarlo rapido en otro problema de malla 2D:

1. reparte la grilla por filas,
2. agrega ghost rows,
3. intercambia fronteras con `Sendrecv`,
4. calcula solo el bloque interno,
5. repite,
6. junta la malla final solo si hace falta.

Esa es la estructura central de `rows subdomain`.
