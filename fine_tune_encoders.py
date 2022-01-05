import os

import torch
import wandb

import torch.nn as nn
import torchvision

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichModelSummary

import yaml

from src.data.utils import (
    load_cub_dataset,
    load_cifar100_dataset,
    load_cifar10_dataset,
    load_cars_dataset,
    load_caltect_dataset
)
from src.models import CifarResNet18, LeNet
from src.utils import flatten_dict

class GenericPlaceHolder(pl.LightningModule):
    def __init__(self, 
                 encoder):
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


class FineTuneModels(pl.LightningModule):
    def __init__(self, 
                 encoder,
                 embedding_size,
                 n_classes,
                 loss_fn):
        super().__init__()

        self.save_hyperparameters()
        
        self.encoder = encoder
        self.loss_func = loss_fn
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(32,32)),
            torchvision.transforms.RandomHorizontalFlip(p=0.3),
            torchvision.transforms.RandomVerticalFlip(p=0.3),
            torchvision.transforms.RandomPerspective(distortion_scale=0.2),
            torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768]),
        ])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768])
        ])

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        
        self.model =nn.Sequential(
          self.encoder,
          nn.ReLU(),
          nn.Linear(embedding_size,n_classes)
        )

    def configure_optimizers(self):        
        params = list(self.parameters())
        optimizer = torch.optim.Adam(params)       
        return optimizer

    def forward(self, image):

        return self.model(image)

    def training_step(self, train_batch, _):
        images, labels = train_batch
        images = self.train_transform(images)
        logits = self(images)
        loss = self.loss_func(logits, labels)
        self.train_accuracy(logits.softmax(dim=-1), labels)
        
        self.log('train_acc', self.train_accuracy, on_epoch=True, on_step=True)
        self.log(f'train_loss_finetune', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, val_batch, _):
        images, labels = val_batch
        images = self.transform(images)
        logits = self(images)
        loss = self.loss_func(logits, labels)
        self.val_accuracy(logits.softmax(dim=-1), labels)
        
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False)
        self.log(f'val_loss_finetune', loss, on_epoch=True, on_step=False)
        return loss
    

class LinearImageClassifier(pl.LightningModule):
    def __init__(self, 
                 encoder,
                 embedding_size,
                 n_classes,
                 loss_fn):
        super().__init__()

        self.save_hyperparameters()
 
        self.encoder = encoder
        self.loss_func = loss_fn
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(32,32)),
            torchvision.transforms.RandomHorizontalFlip(p=0.3),
            torchvision.transforms.RandomVerticalFlip(p=0.3),
            torchvision.transforms.RandomPerspective(distortion_scale=0.2),
            torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768]),
        ])

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768])
        ])
        
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        
        self.linear_head =nn.Sequential(
          nn.Linear(embedding_size,n_classes)
        )

    def configure_optimizers(self):        
        params = list(self.linear_head.parameters())
        # optimizer = torch.optim.Adam(params)       
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        return optimizer

    def forward(self, image):
        with torch.no_grad():
            self.encoder.eval()
            reps = self.encoder(image)
        
        logits = self.linear_head(reps)
        return logits

    def training_step(self, train_batch, _):
        images, labels = train_batch
        images = self.train_transform(images)
        logits = self(images)
        loss = self.loss_func(logits, labels)
        self.train_accuracy(logits.softmax(dim=-1), labels)
        
        self.log('train_acc', self.train_accuracy, on_epoch=True, on_step=True)
        self.log(f'train_loss_finetune', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, val_batch, _):
        images, labels = val_batch
        images = self.transform(images)
        logits = self(images)
        loss = self.loss_func(logits, labels)
        self.val_accuracy(logits.softmax(dim=-1), labels)
        
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False)
        self.log(f'val_loss_finetune', loss, on_epoch=True, on_step=False)
        return loss


def main():

    with open('./config/config_fine_tune.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    loaders = {
        'cifar10': load_cifar10_dataset,
        'cifar100': load_cifar100_dataset,
        'cub': load_cub_dataset,
        'cars196': load_cars_dataset,
        'caltech': load_caltect_dataset
    }

    train, test = loaders[cfg['new_data']]('./')

    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        test,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False
    )


    if cfg['original_data']=='cifar10':
        cnn_encoder = CifarResNet18(
            embedding_dim=cfg['embedding_dim'], 
            hidden_dim=1024,
            logits=True if cfg['need_logits'] else False,
            number_classes=10 if cfg['need_logits'] else None,
        )
    else:
        cnn_encoder = LeNet(
            embedding_dim=cfg['embedding_dim'], 
            logits=True if cfg['need_logits'] else False,
            number_classes=10 if cfg['need_logits'] else None,
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
            loss_fn=nn.CrossEntropyLoss()
        )
    elif cfg['method']=='linear':
        model = LinearImageClassifier(
            encoder=encoder,
            embedding_size=cfg['embedding_dim'],
            n_classes=cfg['n_classes'],
            loss_fn=nn.CrossEntropyLoss()
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
        save_top_k=2, 
        monitor='val_acc',
        save_weights_only=False,
        save_last=True
    )

    trainer = pl.Trainer( 
        logger=wandb_logger,    
        gpus=None if not torch.cuda.is_available() else -1,
        max_epochs=30,           
        deterministic=True, 
        precision=32,   
        profiler="simple",
        callbacks=[checkpoint_callback, RichModelSummary()],
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=test_dataloader
    )    
    wandb.finish()

if __name__ == "__main__":
    main()