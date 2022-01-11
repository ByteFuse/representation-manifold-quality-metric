import os

import pandas as pd
from sklearn import preprocessing

import torch
import torchvision 

from src.data.image import GenericImageSet

def load_caltect_dataset(dir):
    data_dir = os.path.join(dir, "data/101_ObjectCategories/")
    
    caltech_dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform= torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((32, 32))
        ])
    )

    caltech_train, caltech_test = torch.utils.data.random_split(
        caltech_dataset,
        [int(len(caltech_dataset)*0.8), len(caltech_dataset)-int(len(caltech_dataset)*0.8)],
        generator=torch.Generator().manual_seed(42)
    )
    return caltech_train, caltech_test


def load_cars_dataset(dir):
    le = preprocessing.LabelEncoder()

    data_dir = os.path.join(dir, "data/cars/")

    data = pd.read_csv(os.path.join(data_dir,"meta_data.csv"))
    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']

    train_data['class'] = le.fit_transform(train_data['class'])
    test_data['class'] = le.transform(test_data['class'])

    cars_train = GenericImageSet(train_data, data_dir, size=32, min_channels=3)
    cars_test = GenericImageSet(test_data, data_dir, size=32, min_channels=3)

    return cars_train, cars_test


def load_cub_dataset(dir):
    data_dir = os.path.join(dir, "data/cub/CUB_200_2011/")

    images = pd.read_csv(os.path.join(data_dir, 'images.txt'), names=['id', 'image'], sep=' ')
    splits = pd.read_csv(os.path.join(data_dir, 'train_test_split.txt'), names=['id', 'split'], sep=' ')
    classes = pd.read_csv(os.path.join(data_dir, 'image_class_labels.txt'), names=['id', 'class'], sep=' ' )

    data = pd.merge(left=images, right=pd.merge(left=splits, right=classes))
    train_data = data[data['split'] == 1]
    test_data = data[data['split'] == 0]

    cub_train = GenericImageSet(train_data, os.path.join(data_dir, 'images'), size=32, min_channels=3)
    cub_test = GenericImageSet(test_data, os.path.join(data_dir, 'images'), size=32, min_channels=3)

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


def load_fashion_dataset(dir):
    # getting data directory location
    data_dir = os.path.join(dir, "data/fashion/")

    fashion_train = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=False
    )

    fashion_test = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        download=False
    )

    return fashion_train, fashion_test