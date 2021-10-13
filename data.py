import torchvision
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

transforms = {
    "MNIST": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "CIFAR10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
        root="./data", train=True, download=True, transform=transforms[dataset_name]
    )
    lengths = [len(dataset) // workers_n] * workers_n
    return random_split(dataset, lengths)


def generate_dataloader(dataset_name, workers_n, batch_size=64):
    sub_datasets = split_dataset(dataset_name, workers_n)
    train_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in sub_datasets
    ]
    testset = datasets[dataset_name](
        root="./data", train=False, download=True, transform=transforms[dataset_name]
    )
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loaders, test_loader


if __name__ == "__main__":
    test = split_dataset("MNIST", 10)
