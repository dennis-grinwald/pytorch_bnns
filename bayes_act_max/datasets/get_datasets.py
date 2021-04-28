import os

import numpy as np
import torch
import torchvision
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from bayes_act_max.datasets.imagenet_mini_labels import imagenet_mini_labels_dict
from bayes_act_max.datasets.imagenet_labels import imagenet_labels

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




# Tiny-Imagenet
def load_tiny_imagenet(pretrained=True, batch_size=32):

    dataset_dir = '../../data/tiny-imagenet-200/'

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'val', 'images')
    #unlabeled_dir = os.path.join(dataset_dir, 'test', 'images')

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # Pre-calculated mean & std on imagenet:
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # For other datasets, we could just simply use 0.5:
    # norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    print('Preparing dataset ...')
    # Normalization
    norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        if pretrained else torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Normal transformation
    if pretrained:
        train_trans = [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224),
                        transforms.ToTensor()]
        val_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
        test_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]

    else:
        train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        val_trans = [transforms.ToTensor(), norm]
        test_trans = [transforms.ToTensor(), norm]

    trainset = torchvision.datasets.ImageFolder(train_dir,
                                    transform=transforms.Compose(train_trans + [norm]))

    valset = torchvision.datasets.ImageFolder(train_dir,
                                    transform=transforms.Compose(val_trans))

    num_train = len(trainset)
    indices = list(range(num_train))

    # val set size
    split = 10000

    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]


    trainset = torch.utils.data.Subset(trainset, train_idx)
    valset = torch.utils.data.Subset(valset, val_idx)
    testset = torchvision.datasets.ImageFolder(test_dir,
                                    transform=transforms.Compose(val_trans))

    print('Preparing data loaders ...')

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True
    )

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size,
        shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, num_workers=1, pin_memory=True,
        shuffle=False)

    idx_to_class = {i: c for c, i in testset.class_to_idx.items()}
    class_to_name = get_class_name()

    return trainloader, valloader, testloader, testset, idx_to_class, class_to_name

def create_val_img_folder():
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    dataset_dir = '../../data/tiny-imagenet-200/'
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

def get_class_name():
    dataset_dir = '../../data/tiny-imagenet-200/'
    class_to_name = dict()
    fp = open(os.path.join(dataset_dir, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name[words[0]] = words[1].split(',')[0]
    fp.close()
    return class_to_name


def load_imagenet_mini(bs = 32):
    data_dir = 'datasets/imagenet_mini/'

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=True)

    labels_dict = imagenet_mini_labels_dict

    return train_loader, val_loader, labels_dict

def load_imagenet_full(bs = 32):
    data_dir = 'datasets/imagenet/'

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=True)

    labels_list = imagenet_labels

    return train_loader, val_loader, labels_list

def load_places365(bs = 32):
    data_dir = '/home/dgrinwald/data/places365/places365_standard/'

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=True)

    labels_list = imagenet_labels

    return train_loader, val_loader, labels_list


# if __name__ == "__main__":
    # train_loader, val_loader, labels_dict = load_imagenet_mini()
    # train_loader, val_loader, labels_dict = load_imagenet_full()
    # print(train_loader.dataset.shape)
