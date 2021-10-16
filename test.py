import os
import torch
import torch.distributed as dist
from torch import optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torch.nn.functional as F

from data import generate_dataloader
from gar import average
from models import MNIST
from utils import CUDA, collect_grads, set_grads


def run2(rank, size, src, dst, train_loader):
    """Distributed function to be implemented later."""
    print(f"Rank: {rank}")
    # need to pre allocate the memory
    grads = [CUDA(torch.zeros((25450, 1)))] * len(src)
    num_batches = len(train_loader.dataset) // float(64)
    # group = dist.new_group(ranks)
    for epoch in range(100):
        epoch_loss = 0
        for data, target in train_loader:
            data, target = CUDA(Variable(data)), CUDA(Variable(target))
            meta_model = CUDA(MNIST())
            optimizer = optim.Adam(meta_model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            predict_y = meta_model(data)
            loss = F.cross_entropy(predict_y, target)
            epoch_loss += loss.item()
            # get self grad
            grad = collect_grads(meta_model, loss)
            # dist.all_reduce(grad, op=dist.reduce_op.SUM)
            for d in dst:
                dist.send(tensor=grad, dst=d)
                print(f"Rank {rank} send grad to {d}")
            for i, s in enumerate(src):
                dist.recv(tensor=grads[i], src=s)
                print(f"Rank {rank} receive grad from {s}")
            grad = average(grads + [grad])
            set_grads(meta_model, grad)
            optimizer.step()
        print(
            "Rank ", dist.get_rank(), ", epoch ", epoch, ": ", epoch_loss / num_batches
        )


def run(rank, size):
    tensor = CUDA(torch.zeros(1))
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print("Rank ", rank, " has data ", tensor[0])


def init_processes(rank, size, src, dst, fn, train_loader, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, src[rank], dst[rank], train_loader)
    # fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    torch.multiprocessing.set_start_method("spawn")
    train_loaders, test_loader = generate_dataloader("MNIST", size, batch_size=64)
    src = [
        [1, 2, 3, 4],
        [],
        [],
        [],
        [],
    ]
    dst = [[], [0], [0], [0]]
    for rank in range(size):
        p = Process(
            target=init_processes,
            args=(rank, size, src, dst, run2, train_loaders[rank]),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
