import random
from tqdm import tqdm

import torch 

from src.data.transforms import EasyTransformations, MeduimTransformations, HardTransformations


class ImageCentriodMQM():
    
    def __init__(
        self, 
        dataloader, 
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        number_of_runs=10,
        supervised=False):
        
        assert isinstance(dataloader, torch.utils.data.dataloader.DataLoader), 'dataloader must be of type torch.utils.data.dataloader.DataLoader'

        self.augmentation_distributions = {
            'easy':EasyTransformations(image_size=image_size, mean=mean, std=std),
            'meduim': MeduimTransformations(image_size=image_size, mean=mean, std=std),
            'hard':HardTransformations(image_size=image_size, mean=mean, std=std)
        }
        self.dataloader = dataloader  
        self.number_of_runs = number_of_runs
        self.supervised=supervised

    def calculate_mqm(self, representations, labels):
        print(len(representations))
        if self.supervised:
            unique_labels = torch.unique(torch.tensor(labels), sorted=True, return_inverse=False)
            
            centroid_mqm = 0

            for label in unique_labels:
                indices = (torch.tensor(labels) == torch.tensor(label))
                indices = indices.nonzero()
                indices = torch.tensor([int(index[0]) for index in indices])

                centriods = torch.mean(representations[indices], dim=1)
                distance_matrix = torch.cdist(centriods, centriods, p=2)
                centroid_mqm += torch.mean(distance_matrix)
            
            centroid_mqm /= len(unique_labels)

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
                
            image = augmentations(image)
    
            with torch.no_grad():
                representations.append(model(image.to(device)).cpu())

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
            augmented_representations.extend(representations)
        
        centriod_mqm = self.calculate_mqm(augmented_representations, augmented_labels)

        if training_state:
            model.train()

        return centriod_mqm
    
    
    __call__ = forward


class ImagePointWiseMQM(ImageCentriodMQM):
    def __init__(
        self, 
        dataloader, 
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        number_of_runs=10,
        supervised=False):

        super().__init__(dataloader, image_size, mean, std, number_of_runs, supervised)

    def calculate_mqm(self, representations, labels):

        if self.supervised:
            labels = torch.unique(labels, sorted=True, return_inverse=False)
            
            point_wise_mqm = 0

            for label in labels:
                indices = labels == label
                indices = indices.nonzero()

                distance_matrix = torch.cdist(representations, representations, p=2)
                point_wise_mqm += torch.mean(distance_matrix)
            
            point_wise_mqm /= len(labels)

        else:
            distance_matrix = torch.cdist(representations, representations, p=2)
            point_wise_mqm = torch.mean(distance_matrix)
        
        return point_wise_mqm