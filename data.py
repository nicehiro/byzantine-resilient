import torchvision
from torch.utils.data import random_split, Subset
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


def split_dataset(dataset_name, workers_n, split_method=random_split):
    dataset = datasets[dataset_name](
        root="./data",
        train=True,
        download=True,
        transform=train_transforms[dataset_name],
    )
    ave = len(dataset) // workers_n
    lengths = [ave] * (workers_n - 1)
    lengths.append(len(dataset) - ave * (workers_n - 1))
    return split_method(dataset, lengths)


def sequencial_split(dataset, lengths):
    train_datasets = []
    s = e = 0
    for i in range(len(lengths)):
        s = e
        e += lengths[i]
        train_dataset = Subset(dataset, range(s, e))
        train_datasets.append(train_dataset)
    return train_datasets


def custom_split(dataset, lengths):

    def convert_boolean_list_to_indices(boolean_list):
        res = []
        for i in range(len(boolean_list)):
            if boolean_list[i]:
                res.append(i)
        return res

    class_num = len(dataset.classes)
    sorted_indices = []
    # sort dataset with targets
    for c in range(class_num):
        boolean_list = dataset.targets == c
        sorted_indices += convert_boolean_list_to_indices(boolean_list)
    
    train_datasets = []
    s = e = 0
    for i in range(len(lengths)):
        s = e
        e += lengths[i]
        train_dataset = Subset(dataset, sorted_indices[s: e])
        train_datasets.append(train_dataset)
    return train_datasets


def generate_dataloader(dataset_name, workers_n, batch_size=64, split_method=random_split):
    sub_datasets = split_dataset(dataset_name, workers_n, split_method=split_method)
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
