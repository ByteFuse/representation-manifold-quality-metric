import random
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.data.transforms import ExtremelyHardTransformations


class ImageCentriodMQM():
    
    def __init__(
        self, 
        dataloader, 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        number_of_runs=10,
        supervised=False):
        
        assert isinstance(dataloader, torch.utils.data.dataloader.DataLoader), 'dataloader must be of type torch.utils.data.dataloader.DataLoader'

        self.augmentation_distributions = {
            'easy':torchvision.transforms.Compose([
                    torchvision.transforms.RandAugment(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean = (0.1307,), std =  (0.3081,))
            ]),
            'meduim': torchvision.transforms.Compose([
                    torchvision.transforms.TrivialAugmentWide(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean = (0.1307,), std =  (0.3081,))
                ]),
            'hard':ExtremelyHardTransformations(mean=mean, std=std)
        }
        self.dataloader = dataloader  
        self.number_of_runs = number_of_runs
        self.supervised=supervised

    def calculate_centroid_mqm(self, representations, labels):

        if self.supervised:
            labels = torch.unique(labels, sorted=True, return_inverse=False)
            
            centroid_mqm = 0

            for label in labels:
                indices = labels == label
                indices = indices.nonzero()

                centriods = torch.mean(representations[indices], dim=1)
                distance_matrix = torch.cdist(centriods, centriods, p=2)
                centroid_mqm += torch.mean(distance_matrix)
            
            centroid_mqm /= len(labels)

        else:
           centriods = torch.mean(representations, dim=1)
           distance_matrix = torch.cdist(centriods, centriods, p=2)
           centroid_mqm = torch.mean(distance_matrix)
        
        return centroid_mqm

    def generate_representations(self, model, augmentations):
        representations = []
        labels = []
        seed = random.randint(0, 100000)
        device = next(model.parameters()).device

        for batch in self.dataloader:
            if self.supervised:
                image, label = batch
                labels.extend(label)
            else:
                torch.manual_seed(seed)
                image = batch
                
            image = torchvision.transforms.ConvertImageDtype(torch.uint8)(image*255)  
            image = augmentations(image)
    
            with torch.no_grad():
                representations.extend(model(image.to(device)).cpu())

        representations = torch.cat(representations, dim=0)

        if self.supervised:
            return representations, labels
        else:
            return representations

        
    def forward(self, model, difficulty='meduim'):

        assert difficulty in ['easy', 'meduim', 'hard'], 'difficulty must be one of: easy, meduim or hard'

        training_state = False

        if model.training:
            model.eval()
            training_state = True

        augmentation_distributions = self.augmentation_distributions[difficulty]

        augmented_representations = []
        augmented_labels =[]

        for _ in tqdm(range(self.number_of_runs)):

            if self.supervised:
                representations, labels = self.generate_representations(model, augmentation_distributions)
                augmented_labels.extend(labels)
            else:
                representations = self.generate_representations(model, augmentation_distributions)
            augmented_representations.append(representations)
        
        centriod_mqm = self.calculate_centroid_mqm(augmented_representations, augmented_labels)

        if training_state:
            model.train()

        return centriod_mqm
    
    
    __call__ = forward