import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import threading


lock = threading.Lock()


def run(rank, size):
    tensor = torch.zeros(1)
    rec_tensor = torch.zeros(1)

    tensor += rank
    if rank == 0:
        r1 = dist.isend(tensor=tensor, dst=1)
        r2 = dist.irecv(tensor=rec_tensor, src=2)
    elif rank == 1:
        r1 = dist.isend(tensor=tensor, dst=2)
        r2 = dist.irecv(tensor=rec_tensor, src=0)
    else:
        r1 = dist.isend(tensor=tensor, dst=0)
        r2 = dist.irecv(tensor=rec_tensor, src=1)
    r1.wait()
    r2.wait()
    print("Rank ", rank, " has data ", rec_tensor)


def init_processes(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 3
    processes = []
    torch.multiprocessing.set_start_method("spawn")
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
