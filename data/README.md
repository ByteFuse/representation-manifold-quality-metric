# Instuctions

In order to recreate all of the expirements, you will require the MNIST, CIFAR10, CIFAR100, CARS196 and the CUB-200-2011 datasets. You can download these required datasets by running the `download_data.py` file, found in this folder. You can then specify whether to download all of the datasets of just some of them. To get help, run the following command in the root folder, after installing the package with `pip install -e`:

`python data/download_data.py --help`

When you have downloaded all of the datasets, the folder will have the following structure. 

    ├── data
    │   ├── cars 
    │       ├── cars_test
                ├── 000001.jpg 
    │       ├── carts_train
    |          ├── 000001.jpg 
    │       ├── meta_data.csv                            
    │   ├── cub 
    │       ├── CUB_200_2011
    |          ├── attributes
    │          ├── images
    │          ├── bounding_boxes.txt 
    │          ├── classes.txt 
    │          ├── image_class_labels.txt
    │          ├── images.txt
    │          ├── README
    │          ├── train_test_split.tx
    │       ├── attributes.txt         
    │   ├── mnist 
    │       ├── processed
    │           ├── test.pt
    │           ├── training.pt
    │       ├── raw
    │   ├── cifar10 
    │       ├── cifar-10-batches-py
    │   ├── cifar100 
    │       ├── cifar-100-batches-py
