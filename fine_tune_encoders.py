import os

import torch
import wandb

import torch.nn as nn
import torchvision

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

import yaml

from src.data.utils import (
    load_cub_dataset,
    load_cifar100_dataset,
    load_cifar10_dataset,
    load_cars_dataset,
    load_caltect_dataset,
    load_fashion_dataset,
    load_mnist_dataset
)
from src.models import CifarResNet18, LeNet
from src.utils import flatten_dict

def ignore_none_collate_fn(batch):
    '''
    Collate function to be used with torch.Dataloader if 
    you wish for the dataloader to ignore None that is returned 
    from dataset itarator and collect a new sample.
    '''
    batch = list(filter(lambda x : x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class GenericPlaceHolder(pl.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def configure_optimizers(self):        
        pass

    def forward(self, image):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass

class LinearImageClassifier(pl.LightningModule):
    def __init__(self, 
                 encoder,
                 embedding_size,
                 n_classes,
                 original_data,
                 loss_fn,
                 optim='adam',
                 apply_augment=True
                 ):
        super().__init__()

        self.save_hyperparameters()
 
        self.encoder = encoder
        self.loss_func = loss_fn
        self.apply_augment = apply_augment
        self.optim=optim

        if original_data=='cifar10':
            self.train_transform = torchvision.transforms.Compose([
                # torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768]),
            ])

            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768])
            ])
        else:
            self.train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.transform = torchvision.transforms.Compose([
                 torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]) 
            
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        
        self.head =nn.Sequential(
          nn.ReLU(),
          nn.Linear(embedding_size,n_classes),
        )


    def configure_optimizers(self):        
        params = list(self.head.parameters())
        if self.optim=='adam':
            optimizer = torch.optim.Adam(params)     
        else:
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        return optimizer

    def forward(self, image):
        with torch.no_grad():
            self.encoder.eval()
            reps = self.encoder(image)
        
        logits = self.head(reps)
        return logits

    def training_step(self, train_batch, _):
        images, labels = train_batch
        if self.apply_augment:
            images = self.train_transform(images)
        else:
            images = self.transform(images)
        labels=labels.to(torch.int64)    
        logits = self(images)
        loss = self.loss_func(logits, labels)
        self.train_accuracy(logits.softmax(dim=-1), labels)
        
        self.log('train_acc', self.train_accuracy, on_epoch=True, on_step=True)
        self.log(f'train_loss_finetune', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, val_batch, _):
        images, labels = val_batch
        images = self.transform(images)
        labels=labels.to(torch.int64)
        logits = self(images)
        loss = self.loss_func(logits, labels)
        self.val_accuracy(logits.softmax(dim=-1), labels)
        
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False)
        self.log(f'val_loss_finetune', loss, on_epoch=True, on_step=False)
        return loss


class FineTuneModels(LinearImageClassifier):
    def __init__(self, 
                 encoder,
                 embedding_size,
                 n_classes,
                 original_data,
                 loss_fn,
                 optim='adam',
                 apply_augment=True):
        super().__init__(encoder, embedding_size, n_classes, original_data, loss_fn, optim, apply_augment)

        self.save_hyperparameters()
        self.model =nn.Sequential(
          encoder,
          nn.ReLU(),
          nn.Linear(embedding_size,n_classes)
        )
        
        # only fine tune last few layers
        for i, params in enumerate(self.model.parameters()):
            if i<50:
                params.requires_grad = False

    def configure_optimizers(self):        
        params = list(self.parameters())
        
        if self.optim=='adam':
            optimizer = torch.optim.Adam(params, lr=1e-4)     
        else:
            optimizer = torch.optim.SGD(params, lr=1e-4, momentum=0.9)
        return optimizer

    def forward(self, image):
        return self.model(image)


def main():

    with open('./config/config_fine_tune.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    loaders = {
        'cifar10': load_cifar10_dataset,
        'cifar100': load_cifar100_dataset,
        'cub': load_cub_dataset,
        'cars196': load_cars_dataset,
        'caltech': load_caltect_dataset,
        'fashion': load_fashion_dataset,
        'mnist': load_mnist_dataset,
    }

    train, test = loaders[cfg['new_data']]('./')

    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=ignore_none_collate_fn

    )

    test_dataloader = torch.utils.data.DataLoader(
        test,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=6,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=ignore_none_collate_fn
    )


    embedding_dims = cfg['embedding_dim']
    for embedding_dim in embedding_dims:
        for method in ['triplet-supervised', 'triplet', 'triplet-entropy', 'random','nt-xent','cross-entropy', ]:
            cfg['embedding_dim'] = embedding_dim #reassign here for logging purposes
            cfg['original_method'] = method

            if cfg['original_data']=='cifar10':
                cnn_encoder = CifarResNet18(
                    embedding_dim=cfg['embedding_dim'], 
                    hidden_dim=1024,
                    logits=True if method in ['cross-entropy','triplet-entropy'] else False,
                    number_classes=10 if method in ['cross-entropy','triplet-entropy'] else None,
                )
            else:
                cnn_encoder = LeNet(
                    embedding_dim=cfg['embedding_dim'], 
                    logits=True if method in ['cross-entropy','triplet-entropy'] else False,
                    number_classes=10 if method in ['cross-entropy','triplet-entropy'] else None,
                )

            model = GenericPlaceHolder(encoder=cnn_encoder)

            if cfg['original_method']!='random':
                encoder_checkpoint = f"./multirun/data={cfg['original_data']}/{cfg['original_method']}/{cfg['original_optim']}/encoder.embedding_dim={cfg['embedding_dim']}/checkpoints/last.ckpt"
                model = model.load_from_checkpoint(encoder_checkpoint)
                print(f"Loaded encoder from {encoder_checkpoint}")	
            encoder = model.encoder
            encoder.logits=False


            if cfg['method']=='fine_tune':
                model = FineTuneModels(
                    encoder=encoder,
                    embedding_size=cfg['embedding_dim'],
                    n_classes=cfg['n_classes'],
                    loss_fn=nn.CrossEntropyLoss(),
                    optim=cfg['fine_tune_optim'],
                    original_data=cfg['original_data'],
                )
            elif cfg['method']=='linear':
                model = LinearImageClassifier(
                    encoder=encoder,
                    embedding_size=cfg['embedding_dim'],
                    n_classes=cfg['n_classes'],
                    loss_fn=nn.CrossEntropyLoss(),
                    optim=cfg['fine_tune_optim'],
                    original_data=cfg['original_data'],
                )


            # set up wandb and trainers
            wandb.login(key=cfg['secrets']['wandb_key'])
            wandb.init(project='mqm',  config=flatten_dict(cfg))
            code = wandb.Artifact('project-source', type='code')
            code.add_file('./fine_tune_encoders.py')
            wandb.run.use_artifact(code)
            wandb_logger = WandbLogger(project='mqm', config=flatten_dict(cfg))
            

            checkpoint_dir = f'finetuning/old_data={cfg["original_data"]}/model={cfg["original_method"]}/original_optim={cfg["original_optim"]}/new_data={cfg["new_data"]}/method={cfg["method"]}/adam/encoder.embedding_dim={cfg["embedding_dim"]}/batch_size={cfg["batch_size"]}'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir, 
                filename='{epoch}-{val_acc:.2f}', 
                save_top_k=1, 
                monitor='val_acc',
                save_weights_only=False,
                save_last=True
            )

            early_stop = EarlyStopping(
                monitor='val_acc',
                patience=2,
                mode='max',
            )

            trainer = pl.Trainer( 
                logger=wandb_logger,    
                gpus=None if not torch.cuda.is_available() else -1,
                max_epochs=100,           
                deterministic=True, 
                # precision=32 if not torch.cuda.is_available() else 16,   
                profiler="simple",
                callbacks=[checkpoint_callback, RichModelSummary(), early_stop],
            )

            trainer.fit(
                model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=test_dataloader
            )    
            wandb.finish()

if __name__ == "__main__":
    pl.utilities.seed.seed_everything(42)
    main()