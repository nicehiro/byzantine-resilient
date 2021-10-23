import os
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import recv
from torch.multiprocessing import Process
import threading
import logging


lock = threading.Lock()


def run(rank, size):
    tensor = torch.zeros(1)
    recv_list = [torch.zeros_like(tensor) for _ in range(2)]

    tensor += rank
    if rank == 0:
        tensor += 1
        r1 = dist.send(tensor=tensor, dst=1, tag=1)
        tensor += 1
        r2 = dist.send(tensor=tensor, dst=1, tag=2)
    else:
        r1 = dist.recv(tensor=recv_list[0], src=0, tag=1)
        r2 = dist.recv(tensor=recv_list[1], src=0, tag=2)
    logging.critical(f"Rank {rank} has data {recv_list}")


def init_processes(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    torch.multiprocessing.set_start_method("spawn")
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
