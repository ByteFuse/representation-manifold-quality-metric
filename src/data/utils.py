import os

import pandas as pd

import torch
import torchvision 

from src.data.image import GenericImageSet


def load_cars_dataset(dir):
    data_dir = os.path.join(dir, "data/cars/")

    data = pd.read_csv(os.path.join(data_dir,"cars_train.csv"))
    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']

    cars_train = GenericImageSet(train_data, data_dir)
    cars_test = GenericImageSet(test_data, data_dir)

    return cars_train, cars_test


def load_cub_dataset(dir):
    data_dir = os.path.join(dir, "data/cub/CUB_200_2011/")

    images = pd.read_csv(os.path.join(data_dir, 'images.txt'), names=['id', 'image'], sep=' ')
    splits = pd.read_csv(os.path.join(data_dir, 'train_test_split.txt'), names=['id', 'split'], sep=' ')
    classes = pd.read_csv(os.path.join(data_dir, 'image_class_labels.txt'), names=['id', 'class'], sep=' ' )

    data = pd.merge(left=images, right=pd.merge(left=splits, right=classes))
    train_data = data[data['split'] == 1]
    test_data = data[data['split'] == 0]

    cub_train = GenericImageSet(train_data, os.path.join(data_dir, 'images'))
    cub_test = GenericImageSet(test_data, os.path.join(data_dir, 'images'))

    return cub_train, cub_test


def load_cifar100_dataset(dir):

    data_dir = os.path.join(dir, "data/cifar100/")
    cifar100_train = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=True
    )

    cifar100_test = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=True
    )

    return cifar100_train, cifar100_test


def load_cifar10_dataset(dir):

    data_dir = os.path.join(dir, "data/cifar10/")
    cifar10_train = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=True
    )

    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=True
    )

    return cifar10_train, cifar10_test

def load_mnist_dataset(dir):
    # getting data directory location
    data_dir = os.path.join(dir, "data/mnist/")

    mnist_train = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=False
    )


    mnist_test = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=False
    )

    return mnist_train, mnist_test