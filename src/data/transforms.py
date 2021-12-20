import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # required on windows to run albumentations 

import random

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import augly.image as imaugs
import augly.utils as augly_utils

import torch
import torchvision


class EasyTransformations:

    def __init__(
        self,
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        number_transformations = 2,
        ):
        
        self.mean = mean
        self.std = std
        self.numer_transformations = list(range(number_transformations))
        self.image_size=image_size

    def sample_transformations(self):
                
        random.seed()

        possible_transformations = [
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomPerspective(),
        ]

        random_amount_of_augmentations = random.choice(self.numer_transformations)
        return random.sample(possible_transformations, random_amount_of_augmentations)


    def forward(self, image, seed):
        torch.manual_seed(seed)
        transform = torchvision.transforms.Compose(self.sample_transformations())
        second_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_size,self.image_size)),
            torchvision.transforms.Normalize(mean=self.mean,std=self.std)
        ])

        image = second_transform(transform(image))
        return image 


    __call__ = forward


class LocalTransformations:
    """
    Requires numpy as input!
    """

    def __init__(self, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], number_transformations=2):

        self.mean = mean
        self.std = std
        self.numer_transformations = number_transformations
        self.image_size=image_size

    def sample_transformations(self):
                
        random.seed()

        possible_transformations = [
            A.augmentations.transforms.Blur(blur_limit=(3,5)),
            A.augmentations.transforms.CLAHE(),
            A.augmentations.transforms.CoarseDropout(max_holes=3, max_height=3, max_width=3),
            A.augmentations.transforms.Downscale(),
            A.augmentations.transforms.Emboss(),
            A.augmentations.transforms.FancyPCA(),
            A.augmentations.transforms.GaussianBlur(blur_limit=(3,5)),
            A.augmentations.transforms.GaussNoise(),
            A.augmentations.transforms.ISONoise(),
            A.augmentations.transforms.ImageCompression(quality_lower=50),
            A.augmentations.transforms.JpegCompression(quality_lower=50),
            A.augmentations.transforms.RandomBrightness(),
            A.augmentations.transforms.RandomFog(),
            A.augmentations.transforms.RGBShift(),
        ]
        
        return random.sample(possible_transformations, self.numer_transformations)

    def forward(self, image, seed):
        torch.manual_seed(seed)
        transform = A.Compose([A.Compose(self.sample_transformations()), ToTensorV2()])
        second_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_size,self.image_size)),
            torchvision.transforms.Normalize(mean=self.mean,std=self.std)
        ])

        if not isinstance(image, (np.ndarray, np.generic)):
            image = image.numpy()

        image = torch.cat([transform(image=im)['image'].unsqueeze(0)/255 for im in image], dim=0)
        image = second_transform(image)
        return image 
     

    __call__ = forward

class MeduimTransformations:

    def __init__(
        self,
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        number_transformations = 5,
        ):
        
        self.mean = mean
        self.std = std
        self.numer_transformations = list(range(number_transformations))
        self.image_size=image_size

    def sample_transformations(self):
                
        random.seed()
        possible_transformations = [
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomPerspective(),
            torchvision.transforms.GaussianBlur(kernel_size=5),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.RandomGrayscale(),
            torchvision.transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            torchvision.transforms.RandomResizedCrop((self.image_size,self.image_size)),
        ]

        random_amount_of_augmentations = random.choice(self.numer_transformations)
        return random.sample(possible_transformations, random_amount_of_augmentations)


    def forward(self, image,seed):
        torch.manual_seed(seed)
        transform = torchvision.transforms.Compose(self.sample_transformations())
        second_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_size,self.image_size)),
            torchvision.transforms.Normalize(mean=self.mean,std=self.std)
        ])
        image = second_transform(transform(image))
        return image 

    __call__ = forward



class HardTransformations:

    def __init__(
        self,
        distortion_min=0.01,
        distortion_max=0.9,
        perspective_min=1.5,
        perspective_max=45.0,
        colour_changes_min=0.5,
        colour_changes_max=3.0,
        template_options=['web.png', 'mobile.png'],
        number_transformations=5,
        min_probabity_of_applying=0.5,
        max_text_words=10,
        image_size=224,
        meme_options=['lol', 'to the moon!', 'what up doc?'],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ):
        
        self.mean = mean
        self.std = std
        self.distortion_min = distortion_min
        self.distortion_max = distortion_max
        self.perspective_min = perspective_min
        self.perspective_max = perspective_max
        self.colour_changes_min = colour_changes_min
        self.colour_changes_max = colour_changes_max
        self.template_options = template_options
        self.numer_transformations = list(range(number_transformations-2))
        self.min_probabity_of_applying = min_probabity_of_applying
        self.image_size=image_size
        self.max_text_words=max_text_words

        self.augment_image_locations = []
        for root, dirs, files in os.walk(f'{augly_utils.ASSETS_BASE_DIR}/twemojis'):
            for file in files:
                self.augment_image_locations.append(os.path.join(root,file))

        self.font_locations = []
        for root, dirs, files in os.walk(augly_utils.FONT_LIST_PATH.replace('/list', '')):
            for file in files:
                if(file.endswith(".ttf")):
                    self.font_locations.append(os.path.join(root,file))

        self.color_range = list(range(255))
        self.text_choice = list(range(700))
        self.text_max_range = list(range(max_text_words))
        self.meme_options = meme_options
        
    def sample_transformations(self):
                
        random.seed()
    
        distortion = random.uniform(self.distortion_min, self.distortion_max)
        perspective = random.uniform(self.perspective_min, self.perspective_max)
        colour = random.uniform(self.colour_changes_min, self.colour_changes_max)
        template = random.choice(self.template_options)

        possible_transformations = [
            imaugs.Blur(radius=distortion, p=random.uniform(self.min_probabity_of_applying, 1.0)),
            imaugs.Brightness(factor=colour, p=random.uniform(self.min_probabity_of_applying, 1.0)),
            imaugs.ChangeAspectRatio(ratio=perspective, p=random.uniform(self.min_probabity_of_applying, 1.0)),
            imaugs.Contrast(factor=colour, p=random.uniform(self.min_probabity_of_applying, 1.0)),
            imaugs.Grayscale(p=random.uniform(self.min_probabity_of_applying, 1.0)),
            imaugs.ColorJitter(
                brightness_factor=colour,
                contrast_factor=colour, 
                saturation_factor=colour, 
                p=random.uniform(self.min_probabity_of_applying, 1.0)
            ),
            imaugs.OverlayOntoScreenshot(
                template_filepath=os.path.join(augly_utils.SCREENSHOT_TEMPLATES_DIR, template), 
                p=random.uniform(self.min_probabity_of_applying, 1.0)
            ),
            imaugs.ShufflePixels(
                factor=random.uniform(0, 1.0), 
                seed=random.choice([0,42,711,777,9999]), 
                p=random.uniform(self.min_probabity_of_applying, 1.0)
            ),
            imaugs.OverlayImage(
                overlay=random.choice(self.augment_image_locations),
                opacity=random.uniform(0, 1.0),
                overlay_size=random.uniform(0, 1.0),
                x_pos=random.uniform(0, 1.0),
                y_pos=random.uniform(0, 1.0),
                p=random.uniform(self.min_probabity_of_applying, 1.0)
            ),
            imaugs.OverlayText(
                text=random.sample(self.text_choice, random.choice(self.text_max_range)+2),
                font_file=random.choice(self.font_locations), 
                font_size=random.uniform(0, 1.0),
                opacity=random.uniform(0, 1.0),
                color=(random.choice(self.color_range), random.choice(self.color_range), random.choice(self.color_range)),
                x_pos=random.uniform(0, 1.0),
                y_pos=random.uniform(0, 1.0),
                p=random.uniform(self.min_probabity_of_applying, 1.0)
            ),
            imaugs.MemeFormat(
                text=random.choice(self.meme_options),
                font_file=random.choice(self.font_locations), 
                opacity=random.uniform(0, 1.0),
                text_color=(random.choice(self.color_range), random.choice(self.color_range), random.choice(self.color_range)),
                meme_bg_color=(random.choice(self.color_range), random.choice(self.color_range), random.choice(self.color_range)),
                p=random.uniform(self.min_probabity_of_applying, 1.0)
            )
        ]

        random_amount_of_augmentations = random.choice(self.numer_transformations)
        return random.sample(possible_transformations, random_amount_of_augmentations+4)


    def forward(self, image, seed):
    
        if random.random()>0.5:
            transforms = [torchvision.transforms.ToPILImage(), imaugs.Resize(width=256, height=256)]
        else:
            transforms = [torchvision.transforms.RandomResizedCrop((256,256)), torchvision.transforms.ToPILImage()]
        torch.manual_seed(seed)
        transforms.extend(self.sample_transformations())
        first_tranforms = torchvision.transforms.Compose(transforms)

        if random.random()>0.5:
            second_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.image_size,self.image_size)),
                torchvision.transforms.RandomVerticalFlip(0.3),
                torchvision.transforms.RandomHorizontalFlip(0.3),
                torchvision.transforms.Normalize(mean=self.mean,std=self.std)
            ])
        else:
            second_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.image_size,self.image_size)),
                torchvision.transforms.Normalize(mean=self.mean,std=self.std)
            ])


        image = torchvision.transforms.ToTensor()(first_tranforms(image))
        if image.shape[0]>3:
            image = image[:3, :, :]

        return second_transform(image)

    __call__ = forward
