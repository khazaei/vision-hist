import torch
import torchvision
import Imagenette


def CIFAR10(data_set_path, batch_size, transform, train_set, download):
    dataset = torchvision.datasets.CIFAR10(root=data_set_path, train=train_set,
                                           download=download, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    return dataloader


def Imagenet10(data_set_path, batch_size, transform, train_set, download):
    split = 'train' if train_set else 'val'
    dataset = Imagenette.Imagenette(root=data_set_path, transform=transform, split=split, download=download)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=8, pin_memory=True)

    return dataloader
