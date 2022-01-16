import os

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torchvision

import pytorch_lightning as pl

from src.models import LeNet
from src.losses import TripletLoss, TripletLossSupervised, TripletEntropyLoss, NtXentLoss, CrossEntropyLoss
from src.data.utils import load_mnist_dataset
from src.attacks import return_fgsm_contrastive_attack_images, return_fgsm_supervised_attack_images


# EMBEDDING_DIM=256
# OPTIM='sgd'
PGD_ITTERATION=30 #if 1 it is FGSM then alter episilon


class QueryRefrenceImageEncoder(pl.LightningModule):
    def __init__(self, 
                 encoder,
                 loss_fn,
                 optim_cfg,
                 logits=False):
        super().__init__()

        self.save_hyperparameters()
        
        self.encoder = encoder
        self.optim_cfg = optim_cfg
        self.loss_func = loss_fn
        self.logits = logits
        self.val_transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))

    def configure_optimizers(self):        
        pass

    def forward(self, image):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass


if __name__ == "__main__":

    train, test = load_mnist_dataset('../')

    for OPTIM in ['adam']:
        for EMBEDDING_DIM in [128,256,512]:
            encoder = LeNet(
                    embedding_dim=EMBEDDING_DIM, 
                    dropout=0,
                    logits=True,
                    number_classes=10
                )
            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=mnist/triplet-entropy/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_tripent = model.encoder
            encoder_tripent.eval()
            encoder_tripent.cuda()
            encoder_tripent.logits=True



            encoder = LeNet(
                    embedding_dim=EMBEDDING_DIM, 
                    dropout=0,
                    logits=True,
                    number_classes=10
                )
            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=mnist/cross-entropy/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_xent = model.encoder
            encoder_xent.eval()
            encoder_xent.cuda()
            encoder_xent.logits=True



            encoder = LeNet(
                    embedding_dim=EMBEDDING_DIM, 
                    dropout=0,
                    logits=False,
                    number_classes=None
                )

            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=mnist/nt-xent/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_ntxent = model.encoder
            encoder_ntxent.eval()
            encoder_ntxent.cuda()
            encoder_ntxent.logits=False

            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=mnist/triplet-supervised/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_trip_sup = model.encoder
            encoder_trip_sup.eval()
            encoder_trip_sup.cuda()
            encoder_trip_sup.logits=False

            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=mnist/triplet/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_trip = model.encoder
            encoder_trip.eval()
            encoder_trip.cuda()
            encoder_trip.logits=False

            epsilons_dist = np.linspace(start=0, stop=0.2, num=100)

            encoder_random = LeNet(
                    embedding_dim=EMBEDDING_DIM, 
                    dropout=0,
                    logits=False,
                    number_classes=None
                )
            # encoder_random.load_state_dict(torch.load(f'../multirun/mnist_encoder_random_dim{EMBEDDING_DIM}.pt')) #ensure random always the same
            encoder_random.eval()
            encoder_random.cuda()

            models = [encoder_random, encoder_tripent, encoder_xent, encoder_ntxent, encoder_trip_sup, encoder_trip]
            model_names = ['random_init', 'tripent_mnist', 'xent_mnist','ntxent_mnist', 'trip_sup_mnist', 'trip_mnist']
            losses = {
                'random_init': NtXentLoss(),
                'tripent_mnist': TripletEntropyLoss(),
                'xent_mnist': CrossEntropyLoss(),
                'ntxent_mnist': NtXentLoss(),
                'trip_sup_mnist': TripletLossSupervised(),
                'trip_mnist': TripletLoss()
            }

            aug_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))
            dataloader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=False, drop_last=False)

            df = pd.DataFrame()

            for model, name in tqdm(zip(models, model_names), total=len(models), desc='model', leave=False):
                for iteration in tqdm(range(PGD_ITTERATION), desc='pgd iteration', leave=False):
                    projected_points = np.zeros(shape=(1, EMBEDDING_DIM))
                    labels = []

                    for batch in tqdm(dataloader, desc='sample', leave=False):
                        images, labs = batch
                        labels.extend([int(l) for l in labs])

                        if name in ['random_init','ntxent_mnist', 'trip_mnist']:
                            images = return_fgsm_contrastive_attack_images(
                                images=images,
                                model=model,
                                loss_fn=losses[name],
                                val_transform=transform,
                                transform=aug_transform,
                                iterations=iteration,
                                epsilon=2/255 
                            )
                        elif name in ['tripent_mnist', 'xent_mnist']:
                            model.logits = True
                            images = return_fgsm_supervised_attack_images(
                                images=images,
                                model=model,
                                labels=labs,
                                loss_fn=losses[name],
                                final_transform=aug_transform,
                                require_logits=True,
                                iterations=iteration,
                                epsilon=2/255 
                            )
                            model.logits = False
                        else:
                            images = return_fgsm_supervised_attack_images(
                                images=images,
                                model=model,
                                labels=labs,
                                loss_fn=losses[name],
                                final_transform=aug_transform,
                                require_logits=False,
                                iterations=iteration,
                                epsilon=2/255 
                            )
                            model.logits = False

                        with torch.no_grad():
                            
                            reps = model(images.cuda()).cpu().numpy()
                        projected_points = np.concatenate((projected_points, reps))

                    projected_points = projected_points[1:]
                    _ = pd.DataFrame(projected_points)
                    _['pgd_iterations'] = iteration
                    _['model'] = name
                    _['image_index'] = list(range(len(labels)))
                    _['label'] = labels
                    df = pd.concat([df, _])

                fcols = df.select_dtypes('float').columns
                icols = df.select_dtypes('integer').columns

                df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
                df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

                save_loc = f'F://results/data=mnist/{OPTIM}/adverserial_attacks/embedding_dim={EMBEDDING_DIM}/'
                if not os.path.exists(save_loc):
                    os.makedirs(save_loc)
                df.to_pickle(f'{save_loc}/{name}_adverserial.pickle')
                df = pd.DataFrame()
        