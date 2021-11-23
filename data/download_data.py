import argparse
import os

import pandas as pd
import scipy.io

import torchvision

def download_mnist(save_location):
    torchvision.datasets.MNIST(
        root=save_location,
        train=False,
        transform=None,
        target_transform=None,
        download=True
    )

    torchvision.datasets.MNIST(
        root=save_location,
        train=True,
        transform=None,
        target_transform=None,
        download=True
    )


def download_cifar10(save_location):
    torchvision.datasets.CIFAR10(
        root=save_location,
        train=False,
        transform=None,
        target_transform=None,
        download=True
    )

    torchvision.datasets.CIFAR10(
        root=save_location,
        train=True,
        transform=None,
        target_transform=None,
        download=True
    )


def download_cifar100(save_location):
    torchvision.datasets.CIFAR100(
        root=save_location,
        train=False,
        transform=None,
        target_transform=None,
        download=True
    )

    torchvision.datasets.CIFAR100(
        root=save_location,
        train=True,
        transform=None,
        target_transform=None,
        download=True
    )


def download_cars(save_location):
    
    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    os.system(f'curl http://ai.stanford.edu/\~jkrause/car196/cars_train.tgz --output {save_location}/cars_train.tgz')
    os.system(f'curl http://ai.stanford.edu/~jkrause/car196/cars_test.tgz --output {save_location}/cars_test.tgz')
    os.system(f'curl http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz --output {save_location}/car_devkit.tgz')
    os.system(f'curl http://ai.stanford.edu/\~jkrause/car196/cars_test_annos_withlabels.mat --output {save_location}/cars_test_annos_withlabels.mat')

    os.system(f'cd {save_location} && tar -xvf cars_train.tgz && rm cars_train.tgz')
    os.system(f'cd {save_location} && tar -xvf cars_test.tgz && rm cars_test.tgz')
    os.system(f'cd {save_location} && tar -xvf car_devkit.tgz && rm car_devkit.tgz')

    train_meta = scipy.io.loadmat(f'{save_location}/devkit/cars_train_annos.mat')
    test_meta = scipy.io.loadmat(f'{save_location}/devkit/cars_test_annos.mat')
    meta = scipy.io.loadmat(f'{save_location}/devkit/cars_meta.mat')

    jpgs = []
    labels = []
    split = []
    min_x = []
    max_x = []
    min_y = []
    max_y = []

    for met in train_meta['annotations'][0]:
        jpgs.append(f'cars_train/{str(met[-1][0])}')
        labels.append(int(met[-2][0]))
        min_x.append(int(met[0][0]))
        max_x.append(int(met[1][0]))
        min_y.append(int(met[2][0]))
        max_y.append(int(met[3][0]))
        split.append('train')

        
    for met in test_meta['annotations'][0]:
        jpgs.append(f'cars_test/{str(met[-1][0])}')
        labels.append(int(met[-2][0]))
        min_x.append(int(met[0][0]))
        max_x.append(int(met[1][0]))
        min_y.append(int(met[2][0]))
        max_y.append(int(met[3][0]))
        split.append('test')
        
        
    df = pd.DataFrame({
        'image':jpgs,
        'label':labels,
        'min_x': min_x,
        'min_y': min_y,
        'max_x': max_x,
        'max_y': max_y,
        'split':split})

   
    class_names = meta['class_names'][0]
    class_names = [str(c[0]) for c in class_names]
    meta_data = pd.DataFrame({'label':range(0,196), 'class':class_names})
    df = pd.merge(df, meta_data)
    df.to_csv(f'{save_location}/meta_data.csv', index=False)

    os.system(f'cd {save_location} && rm devkit -r')


def download_cub(save_location):

    if not os.path.isdir(save_location):
        os.mkdir(save_location)

    os.system(f'gdown https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45  --output {save_location}/CUB_200_2011.tgz')
    os.system(f'cd {save_location} && tar -xvf CUB_200_2011.tgz && rm CUB_200_2011.tgz')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Scripts to download datasets used in the measuring manfifold quality paper.')
    parser.add_argument('--dataset', type=str, default='all', help='What dataset should be downloaded. Options are all, mnist, cifar10, cifar100, cars, cub.')
    args = parser.parse_args()

    assert args.dataset.lower() in ["all", "mnist", "cifar10", "cifar100", "cars", "cub"], "Only allowed datasets are all, mnist, cifar10, cifar100, cars, cub"

    # getting data directory location
    data_dir = os.path.dirname(__file__)

    mnsit_loc = os.path.join(data_dir, 'mnist')
    cifar10_loc = os.path.join(data_dir, 'cifar10')
    cifar100_loc = os.path.join(data_dir, 'cifar100')
    cars_loc = os.path.join(data_dir, 'cars')
    cub_loc = os.path.join(data_dir, 'cub')

    if args.dataset.lower() == "all":
        print('Downloading MNIST data')
        download_mnist(save_location=mnsit_loc)
        print('Downloading CIFAR10 data')
        download_cifar10(save_location=cifar10_loc)
        print('Downloading CIFAR100 data')
        download_cifar100(save_location=cifar100_loc)
        print('Downloading CARS196 data')
        download_cars(save_location=cars_loc)
        print('Downloading CUB100 data')
        download_cub(save_location=cub_loc)
    elif args.dataset.lower() == 'mnist':
        print('Downloading MNIST data')
        download_mnist(save_location=mnsit_loc)
    elif args.dataset.lower() == 'cifar10':
        print('Downloading CIFAR10 data')
        download_cifar10(save_location=cifar10_loc)
    elif args.dataset.lower() == 'cifar100':
        print('Downloading CIFAR100 data')
        download_cifar100(save_location=cifar100_loc)
    elif args.dataset.lower() == 'cars':
        print('Downloading CARS196 data')
        download_cars(save_location=cars_loc)    
    elif args.dataset.lower() == 'cub':
        print('Downloading CUB100 data')
        download_cub(save_location=cub_loc)   


        



