import torch

class ImageSet(torch.utils.data.Dataset):

    def __init__(self, dataset, transform, labels=True):

        self.dataset = dataset
        self.transform = transform
        self.labels = labels

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        if self.labels:
            image, label = self.dataset[idx]
            image = self.transform(image)
            return image, label
        else:
            image = self.dataset[idx]
            image = self.transform(image)
            return image

class QueryReferenceImageSet(torch.utils.data.Dataset):

    def __init__(self, dataset, transform1, transform2, labels=True):

        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self.labels = labels

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):
        
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