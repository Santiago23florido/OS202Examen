# README - MPI en Python con mpi4py
## Guia completa tipo soporte de curso + ejemplos copiables

Este README reune en un solo documento:

- gestion del entorno MPI,
- comunicaciones punto a punto bloqueantes,
- comunicaciones punto a punto no bloqueantes,
- comunicaciones colectivas,
- variantes vectoriales (`Scatterv`, `Gatherv`, `Allgatherv`, `Alltoallv`),
- reducciones colectivas,
- intercambio total,
- grupos y comunicadores.

Todos los ejemplos estan en Python + mpi4py y estan escritos para que los puedas copiar y pegar directamente.

---

## 1. Instalacion

```bash
pip install mpi4py numpy
```

Ejecutar un programa MPI:

```bash
mpiexec -n 4 python archivo.py
```

## 2. Convencion usada en este README

En este README se usa:

- `numpy` para representar buffers MPI,
- metodos con mayuscula como `Send`, `Recv`, `Bcast`, etc.,
- `MPI.COMM_WORLD` como comunicador principal.

Eso es lo mas parecido al estilo clasico de MPI en C.

Si quieres enviar un valor Python simple en lugar de un `np.array`, `mpi4py` tambien permite usar metodos en minuscula como `send`, `recv`, `bcast` y `gather`.

## 3. Plantilla minima base

```python
from mpi4py import MPI
import numpy as np

if not MPI.Is_initialized():
    MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hola desde rank {rank} de {size}")

if not MPI.Is_finalized():
    MPI.Finalize()
```

## 4. Gestion del entorno MPI

### 4.1 `MPI_Init`, `MPI_Comm_size`, `MPI_Comm_rank`, `MPI_Wtime`, `MPI_Wtick`, `MPI_Finalize`

Archivo sugerido: `env_basic.py`

```python
from mpi4py import MPI
import numpy as np

if not MPI.Is_initialized():
    MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

tick = MPI.Wtick()
t0 = MPI.Wtime()

local_vec = np.arange(rank * 5, rank * 5 + 5, dtype=np.intc)
local_sum = np.sum(local_vec * local_vec)

t1 = MPI.Wtime()

print(f"[rank {rank}/{size}] local_vec={local_vec}, suma_local={local_sum}")
print(f"[rank {rank}] tiempo_local={t1 - t0:.6e} s, wtick={tick:.6e} s")

if not MPI.Is_finalized():
    MPI.Finalize()
```

Ejecucion:

```bash
mpiexec -n 4 python env_basic.py
```

### 4.2 `MPI_Abort`

Archivo sugerido: `env_abort.py`

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("Rank 0 detecto un error fatal. Abortando con codigo 1.")
    comm.Abort(1)
```

Ejecucion:

```bash
mpiexec -n 4 python env_abort.py
```

## 5. Punto a punto bloqueante

### 5.1 `MPI_Send` y `MPI_Recv`

Archivo sugerido: `send_recv.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 10

send_vec = np.array([10, 20, 30, 40], dtype=np.intc)
recv_vec = np.empty(4, dtype=np.intc)

if rank == 0:
    comm.Send([send_vec, MPI.INT], dest=1, tag=tag)
    print(f"[rank 0] enviado -> {send_vec}")

elif rank == 1:
    status = MPI.Status()
    comm.Recv([recv_vec, MPI.INT], source=0, tag=tag, status=status)
    count = status.Get_count(MPI.INT)
    print(f"[rank 1] recibido <- {recv_vec}")
    print(f"[rank 1] source={status.source}, tag={status.tag}, count={count}")
```

### 5.2 `send` y `recv` con un valor unico de Python

Archivo sugerido: `send_recv_scalar.py`

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 11

if rank == 0:
    value = 12345
    comm.send(value, dest=1, tag=tag)
    print(f"[rank 0] enviado valor unico -> {value}")

elif rank == 1:
    value = comm.recv(source=0, tag=tag)
    print(f"[rank 1] recibido valor unico <- {value}")
```

Este estilo es util para enteros, `float`, `str`, listas pequenas y otros objetos Python serializables.

### 5.3 `MPI_Ssend`

Archivo sugerido: `ssend.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 22

send_val = np.array([123], dtype=np.intc)
recv_val = np.empty(1, dtype=np.intc)

if rank == 0:
    comm.Ssend([send_val, MPI.INT], dest=1, tag=tag)
    print(f"[rank 0] Ssend completado. valor enviado={send_val[0]}")

elif rank == 1:
    status = MPI.Status()
    comm.Recv([recv_val, MPI.INT], source=0, tag=tag, status=status)
    print(f"[rank 1] valor recibido={recv_val[0]}")
```

### 5.4 `MPI_Sendrecv`

Archivo sugerido: `sendrecv.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

peer = 1 - rank
status = MPI.Status()

send_vec = np.array([rank, rank + 10, rank + 20], dtype=np.intc)
recv_vec = np.empty(3, dtype=np.intc)

comm.Sendrecv(
    sendbuf=[send_vec, MPI.INT],
    dest=peer,
    sendtag=100,
    recvbuf=[recv_vec, MPI.INT],
    source=peer,
    recvtag=100,
    status=status
)

print(f"[rank {rank}] envie {send_vec} a {peer} y recibi {recv_vec} de {status.source}")
```

### 5.5 `MPI_Probe`

Archivo sugerido: `probe.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 99

if rank == 0:
    send_vec = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    comm.Send([send_vec, MPI.DOUBLE], dest=1, tag=tag)
    print(f"[rank 0] enviado vector de longitud {len(send_vec)}: {send_vec}")

elif rank == 1:
    status = MPI.Status()
    comm.Probe(source=0, tag=tag, status=status)
    count = status.Get_count(MPI.DOUBLE)
    recv_vec = np.empty(count, dtype=np.float64)
    comm.Recv([recv_vec, MPI.DOUBLE], source=status.source, tag=status.tag, status=status)

    print(f"[rank 1] Probe detecto mensaje desde source={status.source}, tag={status.tag}")
    print(f"[rank 1] count={count}")
    print(f"[rank 1] recibido={recv_vec}")
```

### 5.6 `MPI_Get_count`

Archivo sugerido: `getcount_after_recv.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 7

if rank == 0:
    send_vec = np.array([11, 22, 33, 44, 55, 66], dtype=np.intc)
    comm.Send([send_vec, MPI.INT], dest=1, tag=tag)
    print(f"[rank 0] enviado={send_vec}")

elif rank == 1:
    recv_vec = np.empty(6, dtype=np.intc)
    status = MPI.Status()

    comm.Recv([recv_vec, MPI.INT], source=0, tag=tag, status=status)
    count = status.Get_count(MPI.INT)

    print(f"[rank 1] recibido={recv_vec}")
    print(f"[rank 1] count={count}, source={status.source}, tag={status.tag}")
```

## 6. Punto a punto no bloqueante

### 6.1 `MPI_Isend` y `MPI_Irecv`

Archivo sugerido: `isend_irecv.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 50

send_vec = np.array([1, 3, 5, 7], dtype=np.intc) + rank * 100
recv_vec = np.empty(4, dtype=np.intc)

if rank == 0:
    req = comm.Isend([send_vec, MPI.INT], dest=1, tag=tag)
    local_result = np.sum(np.arange(100000, dtype=np.int64))
    req.Wait()
    print(f"[rank 0] Isend completado. enviado={send_vec}, trabajo_local={local_result}")

elif rank == 1:
    req = comm.Irecv([recv_vec, MPI.INT], source=0, tag=tag)
    local_result = np.sum(np.arange(50000, dtype=np.int64))
    req.Wait()
    print(f"[rank 1] Irecv completado. recibido={recv_vec}, trabajo_local={local_result}")
```

### 6.2 `MPI_Issend`

Archivo sugerido: `issend.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 60

send_vec = np.array([9, 8, 7, 6], dtype=np.intc)
recv_vec = np.empty(4, dtype=np.intc)

if rank == 0:
    req = comm.Issend([send_vec, MPI.INT], dest=1, tag=tag)
    print("[rank 0] Issend lanzado")
    req.Wait()
    print(f"[rank 0] Issend confirmado. enviado={send_vec}")

elif rank == 1:
    req = comm.Irecv([recv_vec, MPI.INT], source=0, tag=tag)
    req.Wait()
    print(f"[rank 1] recibido={recv_vec}")
```

### 6.3 `MPI_Test`

Archivo sugerido: `test_single.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 70

if rank == 0:
    send_vec = np.array([100, 200, 300], dtype=np.intc)
    req = comm.Isend([send_vec, MPI.INT], dest=1, tag=tag)
    _ = np.sum(np.arange(200000, dtype=np.int64))
    req.Wait()
    print(f"[rank 0] mensaje enviado={send_vec}")

elif rank == 1:
    recv_vec = np.empty(3, dtype=np.intc)
    req = comm.Irecv([recv_vec, MPI.INT], source=0, tag=tag)

    status = MPI.Status()
    while True:
        done = req.Test(status)
        if done:
            break
        _ = np.sum(np.arange(5000, dtype=np.int64))

    print(f"[rank 1] Test detecto finalizacion. recibido={recv_vec}")
    print(f"[rank 1] source={status.source}, tag={status.tag}")
```

### 6.4 `MPI_Testany`

Archivo sugerido: `testany.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 3:
    raise SystemExit("Este ejemplo requiere exactamente 3 procesos.")

tag = 80

if rank == 0:
    recv_a = np.empty(1, dtype=np.intc)
    recv_b = np.empty(1, dtype=np.intc)

    reqs = [
        comm.Irecv([recv_a, MPI.INT], source=1, tag=tag),
        comm.Irecv([recv_b, MPI.INT], source=2, tag=tag),
    ]

    completed = 0
    while completed < 2:
        index, flag = MPI.Request.Testany(reqs)
        if flag:
            if index == 0:
                print(f"[rank 0] completo req {index}, dato={recv_a[0]} desde rank 1")
            elif index == 1:
                print(f"[rank 0] completo req {index}, dato={recv_b[0]} desde rank 2")
            completed += 1

elif rank == 1:
    data = np.array([111], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()

elif rank == 2:
    data = np.array([222], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()
```

### 6.5 `MPI_Testall`

Archivo sugerido: `testall.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 3:
    raise SystemExit("Este ejemplo requiere exactamente 3 procesos.")

tag = 81

if rank == 0:
    recv_a = np.empty(1, dtype=np.intc)
    recv_b = np.empty(1, dtype=np.intc)

    reqs = [
        comm.Irecv([recv_a, MPI.INT], source=1, tag=tag),
        comm.Irecv([recv_b, MPI.INT], source=2, tag=tag),
    ]

    while True:
        all_done = MPI.Request.Testall(reqs)
        if all_done:
            break
        _ = np.sum(np.arange(10000, dtype=np.int64))

    print(f"[rank 0] recibido de rank 1: {recv_a[0]}")
    print(f"[rank 0] recibido de rank 2: {recv_b[0]}")

elif rank == 1:
    data = np.array([10], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()

elif rank == 2:
    data = np.array([20], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()
```

### 6.6 `MPI_Testsome`

Archivo sugerido: `testsome.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 3:
    raise SystemExit("Este ejemplo requiere exactamente 3 procesos.")

tag = 82

if rank == 0:
    recv_a = np.empty(1, dtype=np.intc)
    recv_b = np.empty(1, dtype=np.intc)

    reqs = [
        comm.Irecv([recv_a, MPI.INT], source=1, tag=tag),
        comm.Irecv([recv_b, MPI.INT], source=2, tag=tag),
    ]

    completed = 0
    seen = set()

    while completed < 2:
        ready = MPI.Request.Testsome(reqs)
        if ready is not None:
            for idx in ready:
                if idx not in seen:
                    seen.add(idx)
                    completed += 1
                    if idx == 0:
                        print(f"[rank 0] req {idx} lista, dato={recv_a[0]} desde rank 1")
                    elif idx == 1:
                        print(f"[rank 0] req {idx} lista, dato={recv_b[0]} desde rank 2")
        else:
            _ = np.sum(np.arange(10000, dtype=np.int64))

elif rank == 1:
    data = np.array([500], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()

elif rank == 2:
    data = np.array([900], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()
```

### 6.7 `MPI_Wait`

Archivo sugerido: `wait_single.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 90

if rank == 0:
    send_vec = np.array([7, 14, 21], dtype=np.intc)
    req = comm.Isend([send_vec, MPI.INT], dest=1, tag=tag)
    _ = np.sum(np.arange(100000, dtype=np.int64))
    req.Wait()
    print(f"[rank 0] Wait del envio completado. enviado={send_vec}")

elif rank == 1:
    recv_vec = np.empty(3, dtype=np.intc)
    req = comm.Irecv([recv_vec, MPI.INT], source=0, tag=tag)
    _ = np.sum(np.arange(30000, dtype=np.int64))
    req.Wait()
    print(f"[rank 1] Wait de la recepcion completado. recibido={recv_vec}")
```

### 6.8 `MPI_Waitany`

Archivo sugerido: `waitany.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 3:
    raise SystemExit("Este ejemplo requiere exactamente 3 procesos.")

tag = 91

if rank == 0:
    recv_a = np.empty(1, dtype=np.intc)
    recv_b = np.empty(1, dtype=np.intc)

    reqs = [
        comm.Irecv([recv_a, MPI.INT], source=1, tag=tag),
        comm.Irecv([recv_b, MPI.INT], source=2, tag=tag),
    ]

    status1 = MPI.Status()
    idx1 = MPI.Request.Waitany(reqs, status=status1)
    print(f"[rank 0] primer Waitany -> idx={idx1}, source={status1.source}")

    status2 = MPI.Status()
    idx2 = MPI.Request.Waitany(reqs, status=status2)
    print(f"[rank 0] segundo Waitany -> idx={idx2}, source={status2.source}")

    print(f"[rank 0] datos finales: recv_a={recv_a[0]}, recv_b={recv_b[0]}")

elif rank == 1:
    data = np.array([1234], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()

elif rank == 2:
    data = np.array([5678], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()
```

### 6.9 `MPI_Waitall`

Archivo sugerido: `waitall_ring.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size < 2:
    raise SystemExit("Este ejemplo requiere al menos 2 procesos.")

prev_rank = (rank - 1) % size
next_rank = (rank + 1) % size

recv_from_prev = np.empty(1, dtype=np.intc)
recv_from_next = np.empty(1, dtype=np.intc)

send_to_prev = np.array([rank], dtype=np.intc)
send_to_next = np.array([rank], dtype=np.intc)

reqs = [
    comm.Irecv([recv_from_prev, MPI.INT], source=prev_rank, tag=1),
    comm.Irecv([recv_from_next, MPI.INT], source=next_rank, tag=2),
    comm.Isend([send_to_next, MPI.INT], dest=next_rank, tag=1),
    comm.Isend([send_to_prev, MPI.INT], dest=prev_rank, tag=2),
]

MPI.Request.Waitall(reqs)

print(
    f"[rank {rank}] recibio {recv_from_prev[0]} de prev={prev_rank} "
    f"y {recv_from_next[0]} de next={next_rank}"
)
```

### 6.10 `MPI_Waitsome`

Archivo sugerido: `waitsome.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 3:
    raise SystemExit("Este ejemplo requiere exactamente 3 procesos.")

tag = 92

if rank == 0:
    recv_a = np.empty(1, dtype=np.intc)
    recv_b = np.empty(1, dtype=np.intc)

    reqs = [
        comm.Irecv([recv_a, MPI.INT], source=1, tag=tag),
        comm.Irecv([recv_b, MPI.INT], source=2, tag=tag),
    ]

    completed = 0
    seen = set()

    while completed < 2:
        ready = MPI.Request.Waitsome(reqs)
        if ready is not None:
            for idx in ready:
                if idx not in seen:
                    seen.add(idx)
                    completed += 1
                    if idx == 0:
                        print(f"[rank 0] Waitsome completo req {idx}, dato={recv_a[0]} desde rank 1")
                    elif idx == 1:
                        print(f"[rank 0] Waitsome completo req {idx}, dato={recv_b[0]} desde rank 2")

elif rank == 1:
    data = np.array([41], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()

elif rank == 2:
    data = np.array([82], dtype=np.intc)
    req = comm.Isend([data, MPI.INT], dest=0, tag=tag)
    req.Wait()
```

### 6.11 `MPI_Iprobe`

Archivo sugerido: `iprobe.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    raise SystemExit("Este ejemplo requiere exactamente 2 procesos.")

tag = 93

if rank == 0:
    send_vec = np.array([1.25, 2.50, 3.75, 5.00], dtype=np.float64)
    _ = np.sum(np.arange(100000, dtype=np.int64))
    comm.Send([send_vec, MPI.DOUBLE], dest=1, tag=tag)
    print(f"[rank 0] enviado={send_vec}")

elif rank == 1:
    status = MPI.Status()

    while True:
        flag = comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        if flag:
            break
        _ = np.sum(np.arange(5000, dtype=np.int64))

    count = status.Get_count(MPI.DOUBLE)
    recv_vec = np.empty(count, dtype=np.float64)
    comm.Recv([recv_vec, MPI.DOUBLE], source=status.source, tag=status.tag)

    print(f"[rank 1] Iprobe detecto mensaje desde source={status.source}, tag={status.tag}")
    print(f"[rank 1] count={count}, recibido={recv_vec}")
```

## 7. Comunicaciones colectivas

### 7.1 Regla fundamental

En una colectiva:

- todos los procesos del comunicador deben llamar la funcion,
- no se usa `tag`,
- si usas un subcomunicador, todos los ranks de ese subcomunicador deben participar.

### 7.2 `MPI_Barrier`

Archivo sugerido: `barrier.py`

```python
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    time.sleep(2)

print(f"[rank {rank}] llego antes de Barrier")
comm.Barrier()
print(f"[rank {rank}] paso Barrier")
```

### 7.3 `MPI_Bcast`

Archivo sugerido: `bcast.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

root = 0
buffer = np.empty(5, dtype=np.intc)

if rank == root:
    buffer[:] = np.array([10, 20, 30, 40, 50], dtype=np.intc)

print(f"[rank {rank}] antes de Bcast -> {buffer}")
comm.Bcast([buffer, MPI.INT], root=root)
print(f"[rank {rank}] despues de Bcast -> {buffer}")
```

### 7.4 `MPI_Scatter`

Archivo sugerido: `scatter.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

root = 0
sendcount = 3
recvcount = 3

sendbuf = None
recvbuf = np.empty(recvcount, dtype=np.intc)

if rank == root:
    sendbuf = np.arange(size * sendcount, dtype=np.intc)
    print(f"[rank {rank}] sendbuf completo = {sendbuf}")

comm.Scatter([sendbuf, sendcount, MPI.INT], [recvbuf, MPI.INT], root=root)

print(f"[rank {rank}] recibio -> {recvbuf}")
```

### 7.5 `MPI_Gather`

Archivo sugerido: `gather.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendcount = 2
recvcount = 2

sendbuf = np.array([rank, rank + 100], dtype=np.intc)

if rank == 0:
    recvbuf = np.empty(size * recvcount, dtype=np.intc)
else:
    recvbuf = None

print(f"[rank {rank}] sendbuf = {sendbuf}")

comm.Gather([sendbuf, MPI.INT], [recvbuf, recvcount, MPI.INT], root=0)

if rank == 0:
    print(f"[rank 0] recvbuf final = {recvbuf}")
```

### 7.6 `gather` con un valor unico de Python

Archivo sugerido: `gather_scalar.py`

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

local_value = 100 + rank
all_values = comm.gather(local_value, root=0)

print(f"[rank {rank}] valor local = {local_value}")

if rank == 0:
    print(f"[rank 0] valores reunidos = {all_values}")
```

En este caso cada proceso envia un valor Python simple y el `root` recibe una lista de Python.

### 7.7 `MPI_Gatherv`

Archivo sugerido: `gatherv.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 4:
    raise SystemExit("Este ejemplo requiere exactamente 4 procesos.")

root = 0

counts = np.array([1, 2, 3, 4], dtype=np.intc)
displs = np.zeros(size, dtype=np.intc)
displs[1:] = np.cumsum(counts[:-1])

sendbuf = np.arange(rank + 1, dtype=np.intc) + rank * 10

if rank == root:
    recvbuf = np.empty(np.sum(counts), dtype=np.intc)
else:
    recvbuf = None

print(f"[rank {rank}] sendbuf = {sendbuf}")

comm.Gatherv([sendbuf, MPI.INT], [recvbuf, counts, displs, MPI.INT], root=root)

if rank == root:
    print(f"[rank 0] recvbuf final = {recvbuf}")
```

### 7.8 `MPI_Allgather`

Archivo sugerido: `allgather.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendcount = 2
recvcount = 2

sendbuf = np.array([rank, rank + 10], dtype=np.intc)
recvbuf = np.empty(size * recvcount, dtype=np.intc)

print(f"[rank {rank}] sendbuf = {sendbuf}")

comm.Allgather([sendbuf, MPI.INT], [recvbuf, recvcount, MPI.INT])

print(f"[rank {rank}] recvbuf = {recvbuf}")
```

## 8. Variantes vectoriales (`...v`)

Estas se usan cuando no todos los procesos envian o reciben la misma cantidad de datos.

### 8.1 `MPI_Scatterv`

Archivo sugerido: `scatterv.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 4:
    raise SystemExit("Este ejemplo requiere exactamente 4 procesos.")

root = 0

counts = np.array([1, 2, 3, 4], dtype=np.intc)
displs = np.zeros(size, dtype=np.intc)
displs[1:] = np.cumsum(counts[:-1])

recvcount = counts[rank]
recvbuf = np.empty(recvcount, dtype=np.intc)

if rank == root:
    sendbuf = np.array([10, 20, 21, 30, 31, 32, 40, 41, 42, 43], dtype=np.intc)
    print(f"[rank 0] sendbuf = {sendbuf}")
else:
    sendbuf = None

comm.Scatterv([sendbuf, counts, displs, MPI.INT], [recvbuf, MPI.INT], root=root)

print(f"[rank {rank}] recibio {recvbuf}")
```

### 8.2 `MPI_Allgatherv`

Archivo sugerido: `allgatherv.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 4:
    raise SystemExit("Este ejemplo requiere exactamente 4 procesos.")

counts = np.array([1, 2, 3, 4], dtype=np.intc)
displs = np.zeros(size, dtype=np.intc)
displs[1:] = np.cumsum(counts[:-1])

sendbuf = np.arange(rank + 1, dtype=np.intc) + rank * 100
recvbuf = np.empty(np.sum(counts), dtype=np.intc)

print(f"[rank {rank}] sendbuf = {sendbuf}")

comm.Allgatherv([sendbuf, MPI.INT], [recvbuf, counts, displs, MPI.INT])

print(f"[rank {rank}] recvbuf = {recvbuf}")
```

### 8.3 `MPI_Alltoallv`

Archivo sugerido: `alltoallv.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 3:
    raise SystemExit("Este ejemplo requiere exactamente 3 procesos.")

if rank == 0:
    sendcounts = np.array([1, 1, 1], dtype=np.intc)
    sendbuf = np.array([10, 11, 12], dtype=np.intc)
elif rank == 1:
    sendcounts = np.array([1, 2, 1], dtype=np.intc)
    sendbuf = np.array([20, 21, 22, 23], dtype=np.intc)
else:
    sendcounts = np.array([2, 1, 1], dtype=np.intc)
    sendbuf = np.array([30, 31, 32, 33], dtype=np.intc)

sdispls = np.zeros(size, dtype=np.intc)
sdispls[1:] = np.cumsum(sendcounts[:-1])

all_sendcounts = comm.allgather(sendcounts)

recvcounts = np.array([all_sendcounts[src][rank] for src in range(size)], dtype=np.intc)
rdispls = np.zeros(size, dtype=np.intc)
rdispls[1:] = np.cumsum(recvcounts[:-1])

recvbuf = np.empty(np.sum(recvcounts), dtype=np.intc)

print(f"[rank {rank}] sendbuf={sendbuf}, sendcounts={sendcounts}")

comm.Alltoallv(
    [sendbuf, sendcounts, sdispls, MPI.INT],
    [recvbuf, recvcounts, rdispls, MPI.INT]
)

print(f"[rank {rank}] recvcounts={recvcounts}, recvbuf={recvbuf}")
```

## 9. Reducciones colectivas

### 9.1 `MPI_Reduce`

Archivo sugerido: `reduce_sum.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

count = 3
sendbuf = np.array([rank + 1, 2 * rank, 1], dtype=np.intc)

if rank == 0:
    recvbuf = np.empty(count, dtype=np.intc)
else:
    recvbuf = None

print(f"[rank {rank}] sendbuf = {sendbuf}")

comm.Reduce(
    [sendbuf, MPI.INT],
    [recvbuf, MPI.INT] if rank == 0 else None,
    op=MPI.SUM,
    root=0
)

if rank == 0:
    print(f"[rank 0] resultado Reduce SUM = {recvbuf}")
```

### 9.2 `MPI_Reduce` con `MPI.MAX`

Archivo sugerido: `reduce_max.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sendbuf = np.array([rank * 2, 10 - rank], dtype=np.intc)

if rank == 0:
    recvbuf = np.empty(2, dtype=np.intc)
else:
    recvbuf = None

comm.Reduce(
    [sendbuf, MPI.INT],
    [recvbuf, MPI.INT] if rank == 0 else None,
    op=MPI.MAX,
    root=0
)

if rank == 0:
    print(f"[rank 0] resultado Reduce MAX = {recvbuf}")
```

### 9.3 `MPI_Allreduce`

Archivo sugerido: `allreduce.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sendbuf = np.array([rank + 1, rank + 2], dtype=np.intc)
recvbuf = np.empty(2, dtype=np.intc)

comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.SUM)

print(f"[rank {rank}] sendbuf={sendbuf}, resultado Allreduce SUM={recvbuf}")
```

### 9.4 `MPI_Reduce_scatter`

Archivo sugerido: `reduce_scatter.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

recvcounts = np.array([1] * size, dtype=np.intc)

sendbuf = np.array([rank + i for i in range(size)], dtype=np.intc)
recvbuf = np.empty(1, dtype=np.intc)

print(f"[rank {rank}] sendbuf = {sendbuf}")

comm.Reduce_scatter(
    [sendbuf, MPI.INT],
    [recvbuf, MPI.INT],
    recvcounts=recvcounts,
    op=MPI.SUM
)

print(f"[rank {rank}] recvbuf despues de Reduce_scatter = {recvbuf}")
```

### 9.5 `MPI_Scan`

Archivo sugerido: `scan.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sendbuf = np.array([rank + 1], dtype=np.intc)
recvbuf = np.empty(1, dtype=np.intc)

comm.Scan([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.SUM)

print(f"[rank {rank}] send={sendbuf[0]}, scan acumulado={recvbuf[0]}")
```

## 10. Intercambio total

### 10.1 `MPI_Alltoall`

Archivo sugerido: `alltoall.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendcount = 1
recvcount = 1

sendbuf = np.array([100 * rank + i for i in range(size)], dtype=np.intc)
recvbuf = np.empty(size, dtype=np.intc)

print(f"[rank {rank}] sendbuf = {sendbuf}")

comm.Alltoall([sendbuf, sendcount, MPI.INT], [recvbuf, recvcount, MPI.INT])

print(f"[rank {rank}] recvbuf = {recvbuf}")
```

## 11. Ejemplo clasico de distribucion de filas

Archivo sugerido: `scatter_rows.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

SIZE = 4
root = 1

if size != SIZE:
    raise SystemExit(f"Este ejemplo requiere exactamente {SIZE} procesos.")

sendcount = SIZE
recvcount = SIZE

recvbuf = np.empty(recvcount, dtype=np.float32)

if rank == root:
    sendbuf = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ], dtype=np.float32)
    print(f"[rank {rank}] matriz completa:\n{sendbuf}")
else:
    sendbuf = None

comm.Scatter([sendbuf, sendcount, MPI.FLOAT], [recvbuf, MPI.FLOAT], root=root)

print(f"[rank {rank}] fila recibida = {recvbuf}")
```

## 12. Grupos y comunicadores

### 12.1 Idea general

- un grupo es un conjunto ordenado de procesos,
- un comunicador es el objeto usado para que esos procesos se comuniquen,
- el comunicador por defecto es `MPI.COMM_WORLD`.

### 12.2 `MPI_Comm_group`, `MPI_Group_incl`, `MPI_Comm_create`

Archivo sugerido: `group_create.py`

```python
from mpi4py import MPI
import numpy as np

world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()

if size < 4:
    raise SystemExit("Este ejemplo requiere al menos 4 procesos.")

selected_ranks = [0, 2, 3]

world_group = world.Get_group()
new_group = world_group.Incl(selected_ranks)
new_comm = world.Create(new_group)

if new_comm != MPI.COMM_NULL:
    new_rank = new_comm.Get_rank()
    new_size = new_comm.Get_size()

    sendbuf = np.array([rank], dtype=np.intc)
    recvbuf = np.empty(1, dtype=np.intc)

    new_comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.SUM)

    print(
        f"[world rank {rank}] pertenece al nuevo comunicador | "
        f"new_rank={new_rank}, new_size={new_size}, suma={recvbuf[0]}"
    )

    new_comm.Free()
else:
    print(f"[world rank {rank}] no pertenece al nuevo comunicador")

new_group.Free()
world_group.Free()
```

### 12.3 `MPI_Comm_split`

Archivo sugerido: `comm_split_even_odd.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

color = 0 if rank % 2 == 0 else 1
key = rank

new_comm = comm.Split(color=color, key=key)

new_rank = new_comm.Get_rank()
new_size = new_comm.Get_size()

sendbuf = np.array([rank], dtype=np.intc)
recvbuf = np.empty(1, dtype=np.intc)

new_comm.Allreduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.SUM)

print(
    f"[world rank {rank}] color={color}, new_rank={new_rank}, "
    f"new_size={new_size}, suma_subgrupo={recvbuf[0]}"
)

new_comm.Free()
```

### 12.4 `MPI_Comm_split` por bloques

Archivo sugerido: `comm_split_blocks.py`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mid = size // 2
color = 0 if rank < mid else 1

new_comm = comm.Split(color=color, key=rank)

new_rank = new_comm.Get_rank()
new_size = new_comm.Get_size()

local = np.array([rank * 10], dtype=np.intc)
global_sub = np.empty(1, dtype=np.intc)

new_comm.Allreduce([local, MPI.INT], [global_sub, MPI.INT], op=MPI.SUM)

print(
    f"[world rank {rank}] bloque={color}, new_rank={new_rank}, "
    f"new_size={new_size}, suma_bloque={global_sub[0]}"
)

new_comm.Free()
```
