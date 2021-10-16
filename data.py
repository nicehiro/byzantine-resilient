import torchvision
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

train_transforms = {
    "MNIST": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "CIFAR10": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "CIFAR100": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
}

test_transforms = {
    "MNIST": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "CIFAR10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "CIFAR100": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
}

datasets = {
    "MNIST": torchvision.datasets.MNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
}


def split_dataset(dataset_name, workers_n):
    dataset = datasets[dataset_name](
        root="./data",
        train=True,
        download=True,
        transform=train_transforms[dataset_name],
    )
    ave = len(dataset) // workers_n
    lengths = [ave] * (workers_n - 1)
    lengths.append(len(dataset) - ave * (workers_n - 1))
    return random_split(dataset, lengths)


def generate_dataloader(dataset_name, workers_n, batch_size=64):
    sub_datasets = split_dataset(dataset_name, workers_n)
    train_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in sub_datasets
    ]
    testset = datasets[dataset_name](
        root="./data",
        train=False,
        download=True,
        transform=test_transforms[dataset_name],
    )
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loaders, test_loader


if __name__ == "__main__":
    test = split_dataset("MNIST", 10)
