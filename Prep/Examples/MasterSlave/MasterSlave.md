# Master-Slave con MPI en Python

Este repositorio te deja dos cosas utiles:

1. Un ejemplo corregido de Mandelbrot con estrategia `master-slave` en [Prep/Examples/MasterSlave/MasterSlave.py](Prep/Examples/MasterSlave/MasterSlave.py).
2. Una guia general para describir, copiar y adaptar el patron a otros algoritmos paralelos.

## Idea del patron

El patron `master-slave` reparte el trabajo asi:

- El `master` no hace el calculo pesado.
- El `master` divide el problema en tareas pequenas o medianas.
- Cada `worker` pide o recibe una tarea, la ejecuta y devuelve el resultado.
- El `master` guarda el resultado y reasigna mas trabajo hasta terminar.

Esto sirve especialmente bien cuando:

- el costo de cada tarea no es uniforme,
- no quieres dejar procesos ociosos,
- puedes dividir el problema en bloques independientes.

En Mandelbrot eso pasa mucho: algunas zonas del plano divergen muy rapido y otras tardan bastante mas.

## La idea de optimizacion que conviene conservar

La estrategia buena del ejemplo es esta:

- dividir en `batches` de filas,
- mandar un batch por worker,
- cuando un worker termina, darle inmediatamente el siguiente batch,
- detenerlo solo cuando ya no quedan filas.

Eso es mejor que repartir filas fijas una sola vez, porque balancea mejor la carga.

## Flujo general

1. El `master` crea la lista logica de tareas.
2. Hace un reparto inicial, una tarea por worker.
3. Cada `worker` calcula y devuelve `(metadata, resultado)`.
4. El `master` inserta ese resultado en la salida final.
5. El `master` reasigna otra tarea al worker libre.
6. Cuando no quedan tareas, el `master` envia `STOP`.

## Plantilla general copiable

Esta version usa `send/recv` de objetos Python porque es la forma mas facil de adaptar rapido a distintos algoritmos.

```python
from mpi4py import MPI

ROOT = 0
WORK_TAG = 10
STOP_TAG = 11
RESULT_TAG = 12


def make_tasks():
    return [
        {"task_id": 0, "payload": ...},
        {"task_id": 1, "payload": ...},
        {"task_id": 2, "payload": ...},
    ]


def compute_task(task):
    payload = task["payload"]
    result = ...
    return {
        "task_id": task["task_id"],
        "result": result,
    }


def merge_result(global_result, worker_result):
    task_id = worker_result["task_id"]
    result = worker_result["result"]
    global_result[task_id] = result


def dispatch_work(comm, worker, tasks, next_task_idx):
    if next_task_idx >= len(tasks):
        comm.send(None, dest=worker, tag=STOP_TAG)
        return next_task_idx, False

    comm.send(tasks[next_task_idx], dest=worker, tag=WORK_TAG)
    return next_task_idx + 1, True


def master(comm, size):
    tasks = make_tasks()
    next_task_idx = 0
    active_workers = 0
    global_result = [None] * len(tasks)

    for worker in range(1, size):
        next_task_idx, has_work = dispatch_work(comm, worker, tasks, next_task_idx)
        if has_work:
            active_workers += 1

    while active_workers > 0:
        status = MPI.Status()
        worker_result = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
        worker = status.source

        merge_result(global_result, worker_result)

        next_task_idx, has_work = dispatch_work(comm, worker, tasks, next_task_idx)
        if not has_work:
            active_workers -= 1

    return global_result


def worker(comm):
    while True:
        status = MPI.Status()
        task = comm.recv(source=ROOT, tag=MPI.ANY_TAG, status=status)

        if status.tag == STOP_TAG:
            break

        worker_result = compute_task(task)
        comm.send(worker_result, dest=ROOT, tag=RESULT_TAG)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        raise SystemExit("Ejecuta con al menos 2 procesos.")

    if rank == ROOT:
        result = master(comm, size)
        print(result)
    else:
        worker(comm)


if __name__ == "__main__":
    main()
```

## Como adaptar la plantilla a otro algoritmo

Solo tienes que definir bien estas tres piezas:

### 1. Unidad de trabajo

Debe ser independiente.

Ejemplos:

- Mandelbrot: un bloque de filas.
- Multiplicacion de matrices: un bloque de filas de `C`.
- Integracion numerica: un subintervalo.
- Monte Carlo: un bloque de muestras.
- Procesamiento de imagen: un bloque de pixels o filas.

### 2. Funcion de calculo

Esa es tu funcion real del algoritmo:

```python
def compute_task(task):
    ...
    return resultado
```

Si cambias de problema, esta funcion cambia. El protocolo `master-slave` casi no cambia.

### 3. Funcion de ensamblado

El `master` debe saber donde insertar cada resultado parcial:

```python
def merge_result(global_result, worker_result):
    ...
```

## Cuando usar `send/recv` y cuando usar `Send/Recv`

Usa `send/recv` si:

- quieres escribir rapido,
- el resultado cabe bien como objeto Python,
- estas estudiando el patron.

Usa `Send/Recv` si:

- trabajas con `numpy`,
- quieres mover buffers grandes,
- quieres evitar el costo de serializacion de objetos Python.

El ejemplo de Mandelbrot corregido usa `Send/Recv` para las matrices numericas y mantiene funciones separadas como:

- `compute_rows(...)`
- `dispatch_work(...)`
- `master(...)`
- `worker(...)`
- `main()`

## Ejemplo concreto: Mandelbrot con master-slave

En este problema cada tarea es:

```python
(start_row, row_count)
```

Cada `worker` recibe un rango de filas, calcula esas filas y devuelve:

- que filas resolvio,
- la submatriz con los valores de convergencia.

### Esqueleto copiable para Mandelbrot

```python
from mpi4py import MPI
import numpy as np

ROOT = 0
WORK_TAG = 100
RESULT_META_TAG = 101
RESULT_DATA_TAG = 102
STOP_TAG = 103


def compute_rows(real_axis, scale_y, start_row, row_count):
    imag_axis = -1.125 + scale_y * np.arange(start_row, start_row + row_count)
    complex_grid = real_axis[:, np.newaxis] + 1j * imag_axis[np.newaxis, :]
    return mandelbrot_convergence(complex_grid)


def dispatch_work(comm, worker, next_row, height, batch_size):
    if next_row >= height:
        payload = np.array([-1, 0], dtype=np.intc)
        comm.Send([payload, MPI.INT], dest=worker, tag=STOP_TAG)
        return next_row, False

    row_count = min(batch_size, height - next_row)
    payload = np.array([next_row, row_count], dtype=np.intc)
    comm.Send([payload, MPI.INT], dest=worker, tag=WORK_TAG)
    return next_row + row_count, True


def master(comm, size, width, height, batch_size):
    image_values = np.empty((width, height), dtype=np.double)
    next_row = 0
    active_workers = 0

    for worker in range(1, size):
        next_row, has_work = dispatch_work(comm, worker, next_row, height, batch_size)
        if has_work:
            active_workers += 1

    while active_workers > 0:
        status = MPI.Status()
        meta = np.empty(2, dtype=np.intc)
        comm.Recv([meta, MPI.INT], source=MPI.ANY_SOURCE, tag=RESULT_META_TAG, status=status)

        worker = status.source
        start_row, row_count = int(meta[0]), int(meta[1])
        block = np.empty((width, row_count), dtype=np.double)
        comm.Recv([block, MPI.DOUBLE], source=worker, tag=RESULT_DATA_TAG)

        image_values[:, start_row:start_row + row_count] = block

        next_row, has_work = dispatch_work(comm, worker, next_row, height, batch_size)
        if not has_work:
            active_workers -= 1

    return image_values


def worker(comm, width, scale_x, scale_y):
    real_axis = -2.0 + scale_x * np.arange(width)

    while True:
        status = MPI.Status()
        meta = np.empty(2, dtype=np.intc)
        comm.Recv([meta, MPI.INT], source=ROOT, tag=MPI.ANY_TAG, status=status)

        if status.tag == STOP_TAG:
            break

        start_row, row_count = int(meta[0]), int(meta[1])
        block = compute_rows(real_axis, scale_y, start_row, row_count)

        result_meta = np.array([start_row, row_count], dtype=np.intc)
        comm.Send([result_meta, MPI.INT], dest=ROOT, tag=RESULT_META_TAG)
        comm.Send([block, MPI.DOUBLE], dest=ROOT, tag=RESULT_DATA_TAG)
```

## Regla practica para elegir `batch_size`

- Si el batch es demasiado chico: hay demasiado trafico MPI.
- Si el batch es demasiado grande: empeora el balance de carga.

Una regla simple para empezar:

- usa entre `8` y `64` filas por batch en problemas 2D,
- prueba y mide,
- si los workers quedan mucho tiempo esperando, baja el batch,
- si hay demasiado overhead de mensajes, subelo.

## Errores tipicos en master-slave

Estos son justo los fallos que mas suelen romper el patron:

- usar el mismo tag para trabajo y parada,
- mezclar `send/recv` con `Send/Recv` sin cuidar el formato,
- esperar que `Recv(...)` devuelva el payload, cuando en realidad llena un buffer,
- meter el bucle principal del `master` dentro del reparto inicial,
- olvidar decrementar `active_workers` al mandar `STOP`,
- devolver resultados sin metadata suficiente para reubicarlos.

## Como ejecutar el ejemplo del repo

Instala dependencias:

```bash
python3 -m pip install mpi4py numpy pillow matplotlib
```

Ejecuta con MPI:

```bash
mpiexec -n 4 python3 Prep/Examples/MasterSlave/MasterSlave.py
```

## Si quieres reutilizar este patron en otro problema

La receta corta es:

1. Decide que bloque de trabajo vas a mandar.
2. Escribe `compute_task(...)` o `compute_rows(...)`.
3. Haz que el worker devuelva metadata + resultado.
4. Haz que el master inserte el resultado donde corresponde.
5. Mantiene el reparto dinamico por batches.

Si respetas esa estructura, puedes reutilizar el mismo esqueleto para muchos algoritmos sin cambiar la idea central.
