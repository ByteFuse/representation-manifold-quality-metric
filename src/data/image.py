import os

from PIL import Image
import torch
import torchvision

class GenericImageSet(torch.utils.data.Dataset):

    def __init__(self, metadata, root_dir, size=32, min_channels=1):

        self.images = metadata['image'].values
        self.labels = metadata['class'].values
        self.images = [os.path.join(root_dir, image) for image in self.images]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((size, size))
        ])
        self.min_channels = min_channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]
        image = self.transform(Image.open(image_path))
        if image.size(0) != self.min_channels:
            return None
        label = self.labels[idx]

        return image, label


class ImageSet(torch.utils.data.Dataset):

    def __init__(self, dataset, transform, labels=True):

        self.dataset = dataset
        self.transform = transform
        self.labels = labels

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            if self.labels:
                image, label = self.dataset[idx]
                image = self.transform(image)
                return image, label
            else:
                image = self.dataset[idx]
                image = self.transform(image)
                return image
        except:
            return None

class QueryReferenceImageSet(torch.utils.data.Dataset):

    def __init__(self, dataset, transform1, transform2, labels=True):

        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self.labels = labels

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            if self.labels:
                image, label = self.dataset[idx]
                image1 = self.transform1(image)
                image2 = self.transform2(image)
                return image1, image2, label
            else:
                image = self.dataset[idx]
                image1 = self.transform1(image)
                image2 = self.transform2(image)
                return image1, image2
        except:
            return None