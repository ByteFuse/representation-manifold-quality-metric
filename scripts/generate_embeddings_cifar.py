import os

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torchvision

import pytorch_lightning as pl

from src.models import CifarResNet18

from src.data.utils import load_cifar10_dataset

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
        self.val_transform = torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768])

    def configure_optimizers(self):        
        pass

    def forward(self, image):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass


if __name__ == "__main__":

    train, test = load_cifar10_dataset('../')

    for OPTIM in ['sgd', 'adam']:
        for EMBEDDING_DIM in [16,32,64,128,256,512]:

            encoder = CifarResNet18(
                    embedding_dim=EMBEDDING_DIM, 
                    hidden_dim=1024,
                    logits=True,
                    number_classes=10
                )
            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=cifar10/triplet-entropy/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_tripent = model.encoder
            encoder_tripent.eval()
            encoder_tripent.cuda()
            encoder_tripent.logits=False



            encoder = CifarResNet18(
                    embedding_dim=EMBEDDING_DIM, 
                    hidden_dim=1024,
                    logits=True,
                    number_classes=10
                )
            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=cifar10/cross-entropy/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_xent = model.encoder
            encoder_xent.eval()
            encoder_xent.cuda()
            encoder_xent.logits=False



            encoder = CifarResNet18(
                    embedding_dim=EMBEDDING_DIM, 
                    hidden_dim=1024,
                    logits=False,
                    number_classes=None
                )

            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=cifar10/nt-xent/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_ntxent = model.encoder
            encoder_ntxent.eval()
            encoder_ntxent.cuda()
            encoder_ntxent.logits=False

            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=cifar10/triplet-supervised/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_trip_sup = model.encoder
            encoder_trip_sup.eval()
            encoder_trip_sup.cuda()
            encoder_trip_sup.logits=False

            model = QueryRefrenceImageEncoder(encoder=encoder,loss_fn=None,optim_cfg=None,)
            model = model.load_from_checkpoint(f'../multirun/data=cifar10/triplet/{OPTIM}/encoder.embedding_dim={EMBEDDING_DIM}/checkpoints/last.ckpt')
            model.eval()
            encoder_trip = model.encoder
            encoder_trip.eval()
            encoder_trip.cuda()
            encoder_trip.logits=False


            epsilons_dist = np.linspace(start=0, stop=1.0, num=100)

            encoder_random = CifarResNet18(
                    embedding_dim=EMBEDDING_DIM, 
                    hidden_dim=1024,
                    logits=False,
                    number_classes=None
                )
            encoder_random.load_state_dict(torch.load(f'../multirun/cifar_encoder_random_dim{EMBEDDING_DIM}.pt')) #ensure random always the same
            encoder_random.eval()
            encoder_random.cuda()

            models = [encoder_random, encoder_tripent, encoder_xent,
                    encoder_ntxent, encoder_trip_sup, encoder_trip]
            model_names = ['random_init', 'tripent_cifar10', 'xent_cifar10','ntxent_cifar10', 'trip_sup_cifar10', 'trip_cifar10']



            transform = torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768])
            dataloader = torch.utils.data.DataLoader(test, batch_size=1024, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=False, drop_last=False)

            df = pd.DataFrame()

            for model, name in tqdm(zip(models, model_names), total=len(models), desc='model', leave=False):
                for epsilon in tqdm(epsilons_dist, desc='epsilon', leave=False):
                    projected_points = np.zeros(shape=(1, EMBEDDING_DIM))
                    labels = []

                    for batch in tqdm(dataloader, desc='sample', leave=False):
                        images, labs = batch
                        labels.extend([int(l) for l in labs])
                        images = transform(torch.clip(images + torch.normal(0, epsilon, size=images.shape), 0, 1))

                        with torch.no_grad():
                            reps = model(images.cuda()).cpu().numpy()
                        projected_points = np.concatenate((projected_points, reps))

                    projected_points = projected_points[1:]
                    _ = pd.DataFrame(projected_points)
                    _['epsilon'] = epsilon
                    _['model'] = name
                    _['image_index'] = list(range(len(labels)))
                    _['label'] = labels
                    df = pd.concat([df, _])

                fcols = df.select_dtypes('float').columns
                icols = df.select_dtypes('integer').columns

                df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
                df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

                save_loc = f'../results/data=cifar10/{OPTIM}/embedding_dim={EMBEDDING_DIM}'
                if not os.path.exists(save_loc):
                    os.makedirs(save_loc)
                df.to_pickle(f'{save_loc}/{name}_white_noise_run{0}.pickle')
                df = pd.DataFrame()
        