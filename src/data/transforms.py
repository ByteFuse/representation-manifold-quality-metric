import os
import random

import augly.image as imaugs
import augly.utils as augly_utils

import torchvision


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
        ]

        random_amount_of_augmentations = random.choice(self.numer_transformations)
        return random.sample(possible_transformations, random_amount_of_augmentations)

    def augment_text(self):
        pass

    def forward(self, image):
    
        transform = torchvision.transforms.Compose(self.sample_transformations())
        second_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_size,self.image_size)),
            torchvision.transforms.Normalize(mean=self.mean,std=self.std)
        ])


        return second_transform(transform(image))

    __call__ = forward


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

    def augment_text(self):
        pass

    def forward(self, image):
    
        transform = torchvision.transforms.Compose(self.sample_transformations())
        second_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.image_size,self.image_size)),
            torchvision.transforms.Normalize(mean=self.mean,std=self.std)
        ])


        return second_transform(transform(image))

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

    def augment_text(self):
        pass

    def forward(self, image):
    
        if random.random()>0.5:
            transforms = [torchvision.transforms.ToPILImage(), imaugs.Resize(width=256, height=256)]
        else:
            transforms = [torchvision.transforms.RandomResizedCrop((256,256)), torchvision.transforms.ToPILImage()]
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
