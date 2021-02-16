import torch
import torchvision
from torchvision import transforms

def load_cifar10_big_augmented(batch_size=50):

    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_test_transform = transforms.Compose(
        [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                        transform=train_transform,
                                        download=True)

    trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size, num_workers=4, pin_memory=True,
                    shuffle=True) 

    testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                       download=True, transform=val_test_transform)

    testloader = torch.utils.data.DataLoader(
                    testset, batch_size=batch_size, num_workers=4, pin_memory=True,
                    shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    return trainloader, testloader, trainset, testset, classes


def load_cifar10_small_standard(batch_size=50):

    train_transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_test_transform = transforms.Compose(
        [   transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                        transform=train_transform,
                                        download=True)

    trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size, num_workers=4, pin_memory=True,
                    shuffle=True) 

    testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                       download=True, transform=val_test_transform)

    testloader = torch.utils.data.DataLoader(
                    testset, batch_size=batch_size, num_workers=4, pin_memory=True,
                    shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    return trainloader, testloader, trainset, testset, classes


def load_cifar10_small_augmented(batch_size=50):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                    transform=train_transform,
                    download=True)

    trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size, num_workers=4, pin_memory=True,
                    shuffle=True) 

    testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                    download=True, transform=test_transform)

    testloader = torch.utils.data.DataLoader(
                    testset, batch_size=batch_size, num_workers=4, pin_memory=True,
                    shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    return trainloader, testloader, trainset, testset, classes