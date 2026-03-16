# Bucket Sort con MPI en Python

Este documento acompana el ejemplo de [BucketSort.py](./BucketSort.py) y te deja una version explicable, reusable y copiable del patron de `bucket sort` paralelo.

## Idea general

El `bucket sort` paralelo no trabaja como `master-slave`.

La estrategia correcta que conviene conservar es esta:

1. repartir el vector inicial entre procesos,
2. ordenar localmente,
3. tomar muestras locales,
4. construir `splitters` globales,
5. redistribuir los datos por buckets con `Alltoallv`,
6. ordenar localmente el bucket final de cada rank,
7. reunir el resultado.

Eso funciona bien porque cada proceso termina siendo responsable de un rango de valores.

## Cuando conviene usar bucket sort

Conviene cuando:

- quieres ordenar muchos datos numericos,
- puedes repartirlos entre procesos,
- la distribucion no es extremadamente sesgada,
- puedes estimar buenos `splitters`.

Si los datos estan muy desbalanceados, los buckets pueden quedar desiguales y perder rendimiento.

## Flujo del algoritmo

1. `Scatterv` reparte el vector global.
2. Cada rank hace `sort()` local.
3. Cada rank toma una muestra regular de su bloque ordenado.
4. El root junta todas las muestras.
5. El root ordena esas muestras y elige `size - 1` splitters.
6. `Bcast` comparte esos splitters.
7. Cada rank corta su vector local con `searchsorted`.
8. `Alltoallv` envia cada subrango al rank propietario de ese bucket.
9. Cada rank vuelve a ordenar su bucket local.
10. `Gatherv` puede reconstruir el vector final global en el root.

## La idea de optimizacion que conviene conservar

La optimizacion central del ejemplo es:

- distribuir el trabajo en paralelo desde el inicio,
- usar `sort()` local sobre bloques pequenos,
- usar muestreo regular para aproximar buenos splitters,
- mover solo los datos necesarios a su bucket final.

No hace falta un `master` repartiendo tareas una por una. Aqui el rendimiento viene de las colectivas MPI y de reducir el trabajo de mezcla global.

## Plantilla general copiable

```python
from mpi4py import MPI
import numpy as np

ROOT = 0


def build_counts_displs(total_size, size):
    counts = np.array(
        [total_size // size + (1 if rank < total_size % size else 0) for rank in range(size)],
        dtype=np.intc,
    )
    displs = np.zeros(size, dtype=np.intc)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


def scatter_data(comm, rank, size, global_data):
    if rank == ROOT:
        total_size = global_data.size
        dtype = global_data.dtype
    else:
        total_size = None
        dtype = np.float64

    total_size = comm.bcast(total_size, root=ROOT)
    counts, displs = build_counts_displs(total_size, size)
    local_data = np.empty(counts[rank], dtype=dtype)

    comm.Scatterv(
        [global_data, counts, displs, MPI.DOUBLE],
        [local_data, MPI.DOUBLE],
        root=ROOT,
    )
    return local_data


def local_regular_sample(local_data, size):
    if local_data.size == 0:
        return np.zeros(size, dtype=np.float64)

    sample_idx = np.linspace(0, local_data.size - 1, size, dtype=int)
    return local_data[sample_idx]


def choose_splitters(comm, rank, size, local_sample):
    if rank == ROOT:
        gathered = np.empty(size * local_sample.size, dtype=np.float64)
    else:
        gathered = None

    comm.Gather([local_sample, MPI.DOUBLE], [gathered, local_sample.size, MPI.DOUBLE], root=ROOT)

    if rank == ROOT:
        gathered.sort()
        splitter_idx = np.linspace(0, gathered.size - 1, size + 1, dtype=int)[1:-1]
        splitters = gathered[splitter_idx]
    else:
        splitters = np.empty(size - 1, dtype=np.float64)

    comm.Bcast([splitters, MPI.DOUBLE], root=ROOT)
    return splitters


def redistribute_by_bucket(comm, size, local_data, splitters):
    cuts = np.searchsorted(local_data, splitters, side="right")
    starts = np.concatenate(([0], cuts)).astype(np.intc)
    ends = np.concatenate((cuts, [local_data.size])).astype(np.intc)
    sendcounts = (ends - starts).astype(np.intc)
    sdispls = np.zeros(size, dtype=np.intc)
    sdispls[1:] = np.cumsum(sendcounts[:-1])

    recvcounts = np.empty(size, dtype=np.intc)
    comm.Alltoall([sendcounts, MPI.INT], [recvcounts, MPI.INT])

    rdispls = np.zeros(size, dtype=np.intc)
    rdispls[1:] = np.cumsum(recvcounts[:-1])
    recvbuf = np.empty(np.sum(recvcounts), dtype=np.float64)

    comm.Alltoallv(
        [local_data, sendcounts, sdispls, MPI.DOUBLE],
        [recvbuf, recvcounts, rdispls, MPI.DOUBLE],
    )

    recvbuf.sort()
    return recvbuf


def gather_final(comm, rank, local_data):
    local_count = local_data.size
    counts = comm.gather(local_count, root=ROOT)

    if rank == ROOT:
        counts = np.array(counts, dtype=np.intc)
        displs = np.zeros(counts.size, dtype=np.intc)
        displs[1:] = np.cumsum(counts[:-1])
        global_sorted = np.empty(np.sum(counts), dtype=np.float64)
    else:
        global_sorted = None
        displs = None
        counts = None

    comm.Gatherv(
        [local_data, MPI.DOUBLE],
        [global_sorted, counts, displs, MPI.DOUBLE],
        root=ROOT,
    )
    return global_sorted
```

## Version corta del `main`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == ROOT:
    rng = np.random.default_rng(12345)
    global_data = rng.uniform(0.0, 1.0, size=1_000_000).astype(np.float64)
else:
    global_data = None

local_data = scatter_data(comm, rank, size, global_data)
local_data.sort()

local_sample = local_regular_sample(local_data, size)
splitters = choose_splitters(comm, rank, size, local_sample)
local_data = redistribute_by_bucket(comm, size, local_data, splitters)
global_sorted = gather_final(comm, rank, local_data)

if rank == ROOT:
    print(np.all(global_sorted[:-1] <= global_sorted[1:]))
```

## Como adaptar este bucket sort a otros casos

Puedes reutilizar la misma estructura si cambias:

- el tipo de dato: `float64`, `int32`, `int64`,
- la forma de generar datos,
- la forma de definir splitters si no quieres usar muestreo regular,
- la comparacion, si ordenas por una clave.

Si quieres ordenar estructuras mas complejas, lo normal es:

- ordenar por una clave numerica,
- mover indices o pares `(key, value)`,
- o convertir temporalmente a arrays `numpy` que puedas comunicar con MPI.

## Errores tipicos en bucket sort paralelo

Estos son los fallos mas comunes:

- usar `MPI.INT` con datos `float64`,
- elegir `size` splitters en vez de `size - 1`,
- intentar hacer el `Gatherv` final con los counts iniciales,
- no volver a ordenar despues de `Alltoallv`,
- elegir muestras locales sobre datos que todavia no estan ordenados.

## Archivo corregido del repo

El ejemplo corregido mantiene funciones separadas para que se vea claro el flujo:

- `static_dist(...)`
- `local_regular_sample(...)`
- `box_distribution(...)`
- `box_organize(...)`
- `gather_sorted_vector(...)`
- `main()`

Archivo: [BucketSort.py](./BucketSort.py)

## Como ejecutar el ejemplo

Instala dependencias:

```bash
python3 -m pip install mpi4py numpy
```

Ejecuta:

```bash
mpiexec -n 4 python3 Prep/Examples/BucketSort/BucketSort.py
```

## Resumen practico

Si quieres replicarlo rapido en otro algoritmo de ordenamiento distribuido:

1. reparte los datos,
2. ordena localmente,
3. calcula splitters globales,
4. redistribuye por rangos,
5. vuelve a ordenar localmente,
6. junta el resultado final si hace falta.

Esa es la estructura que debes conservar.
