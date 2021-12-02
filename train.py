import os

import hydra
from omegaconf import DictConfig

import torch
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichModelSummary

import wandb

from src.data.image import ImageSet, QueryReferenceImageSet
from src.data.utils import load_cub_dataset, load_cifar100_dataset, load_cifar10_dataset, load_cars_dataset
from src.losses import TripletLoss, TripletLossSupervised, TripletEntropyLoss, NtXentLoss
from src.models import CifarResNet18, ResnetImageEncoder
from src.utils import plot_embeddings_unimodal, flatten_dict


class TrainerQueryRefrenceSet(pl.LightningDataModule):
 
    def __init__(
        self,
        transform1,
        transform2,
        val_transform,
        dataset_name,
        batch_size=64,
        num_workers=0):
        super().__init__()

        self.transform1 = transform1
        self.transform2 = transform2

        self.val_transform = val_transform

        self.batch_size = batch_size
        self.num_workers = num_workers

        loaders = {
            'cifar10': load_cifar10_dataset,
            'cifar100': load_cifar100_dataset,
            'cub': load_cub_dataset,
            'cars196': load_cars_dataset
        }
        self.loader = loaders[dataset_name]

    def setup(self, stage=None):

        train, test = self.loader()

        self.train_data = QueryReferenceImageSet(
            train, 
            self.transform1,
            self.transform2,
            labels=True
        )

        self.val_data = QueryReferenceImageSet(
            test, 
            self.val_transform,
            self.transform2,
            labels=True
        )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last = True,
            pin_memory=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last = True,
            pin_memory=True,
            num_workers=self.num_workers
        )



class TrainerSingleImageSet(pl.LightningDataModule):
 
    def __init__(
        self,
        transform,
        val_transform,
        dataset_name,
        batch_size=64,
        num_workers=0):
        super().__init__()

        self.transform = transform

        self.val_transform = val_transform

        self.batch_size = batch_size
        self.num_workers = num_workers

        loaders = {
            'cifar10': load_cifar10_dataset,
            'cifar100': load_cifar100_dataset,
            'cub': load_cub_dataset,
            'cars196': load_cars_dataset
        }
        self.loader = loaders[dataset_name]

    def setup(self, stage=None):
        train, test = self.loader(os.path.dirname(__file__))

        self.train_data = ImageSet(
            train, 
            self.transform,
            labels=True
        )

        self.val_data = ImageSet(
            test, 
            self.val_transform,
            labels=True
        )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last = True,
            pin_memory=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last = True,
            pin_memory=True,
            num_workers=self.num_workers
        )


class QueryRefrenceImageEncoder(pl.LightningModule):
    def __init__(self, 
                 encoder,
                 loss_fn,
                 optim_cfg,
                 ):
        super().__init__()

        self.save_hyperparameters()
        
        self.encoder = encoder
        self.optim_cfg = optim_cfg
        self.loss_func = loss_fn
        self.val_transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))

    def configure_optimizers(self):        
        params = list(self.parameters())

        if self.optim_cfg.name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay)
        if self.optim_cfg.name == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay, momentum=self.optim_cfg.momentum, nesterov=self.optim_cfg.nesterov)
           
        return optimizer

    def forward(self, ref_image, query_image):

        ref_emb = self.encoder(ref_image)
        qeury_emb = self.encoder(query_image)
        return ref_emb, qeury_emb

        
    def training_step(self, train_batch, batch_idx):
        images_1, images_2, labels = train_batch
        
        ref_emb, qeury_emb = self(images_1, images_2)
        loss = self.loss_func(ref_emb, qeury_emb)

        self.log(f'train/loss', loss, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        images_1, images_2, labels = val_batch

        ref_emb, qeury_emb = self(images_1, images_2)
        loss = self.loss_func(ref_emb, qeury_emb)

        self.log('valid/loss', loss, on_epoch=True, on_step=False)
        self.log('valid_loss', loss, on_epoch=True, on_step=False)

        return [ref_emb, labels]

    def validation_epoch_end(self, plot_data):
        fig = plot_embeddings_unimodal(plot_data, self.current_epoch, return_fig=True)
        self.logger.experiment.log({
            "val/embeddings": wandb.Image(fig),
            "global_step": self.global_step
            })


class ImageEncoder(pl.LightningModule):
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
        params = list(self.parameters())

        if self.optim_cfg.name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay)
        if self.optim_cfg.name == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay, momentum=self.optim_cfg.momentum, nesterov=self.optim_cfg.nesterov)
           
        return optimizer

    def forward(self, image):

        if self.logits:
            emb, logits =  self.encoder(image)
            return emb, logits
        else:
            return self.encoder(image)

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        
        if self.logits:
            emb, logits =  self(images)
            loss = self.loss_func(emb, logits, labels)
        else:
            emb = self(images)
            loss = self.loss_func(emb, labels)

        self.log(f'train/loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch

        if self.logits:
            emb, logits =  self(images)
            loss = self.loss_func(emb, logits, labels)
        else:
            emb = self(images)
            loss = self.loss_func(emb, labels)

        self.log('valid/loss', loss, on_epoch=True, on_step=False)
        self.log('valid_loss', loss, on_epoch=True, on_step=False)

        return [emb, labels]

    def validation_epoch_end(self, plot_data):
        fig = plot_embeddings_unimodal(plot_data, self.current_epoch, return_fig=True)
        self.logger.experiment.log({
            "val/embeddings": wandb.Image(fig),
            "global_step": self.global_step
            })


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    pl.utilities.seed.seed_everything(42)

    # setup augmentations
    transform1 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg.data.resize,cfg.data.resize)),
        torchvision.transforms.RandomCrop((cfg.data.size,cfg.data.size)),
        torchvision.transforms.GaussianBlur(cfg.data.blur_kernel),
        torchvision.transforms.RandomRotation(cfg.data.rotation),
        torchvision.transforms.Normalize(cfg.data.mean, cfg.data.std),
    ])

    transform2 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg.data.size,cfg.data.size)),
        torchvision.transforms.RandomPerspective(0.5, 0.8),
        torchvision.transforms.Normalize(cfg.data.mean, cfg.data.std)
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg.data.resize,cfg.data.resize)),
        torchvision.transforms.RandomCrop((cfg.data.size,cfg.data.size)),
        torchvision.transforms.GaussianBlur(cfg.data.blur_kernel),
        torchvision.transforms.RandomPerspective(0.5, 0.5),
        torchvision.transforms.Normalize(cfg.data.mean, cfg.data.std)
    ])
    
    # setup encoder
    if 'cifar' in cfg.data.name:
        encoder = CifarResNet18(
            embedding_dim=cfg.encoder.embedding_dim, 
            logits=cfg.encoder.logits,
            number_classes=cfg.encoder.number_classes
        )
    else:
        encoder = ResnetImageEncoder(
            embedding_dim=cfg.encoder.embedding_dim, 
            resnet_size=18,
            pretrained=cfg.encoder.pretrained,
            logits=cfg.encoder.logits,
            number_classes=cfg.encoder.number_classes
        )

    # setup loss
    if cfg.loss.name == 'triplet':
        loss = TripletLoss(cfg.loss.margin)
    elif cfg.loss.name == 'triplet-supervised':
        loss = TripletLossSupervised(cfg.loss.margin)
    elif cfg.loss.name == 'triplet-entropy':
        loss = TripletEntropyLoss(
            margin=cfg.loss.margin,
            cel_weigth=cfg.loss.cel_weight,
            te_weight=cfg.loss.tel_weight
        )
    elif cfg.loss.name == 'nt-xent':
        loss = NtXentLoss(temperature=cfg.loss.temperature)

    # setup dataset
    if cfg.data.dataset == 'query-reference':
        data = TrainerQueryRefrenceSet(
            transform1=transform1, 
            transform2=transform2, 
            batch_size=cfg.data.batch_size, 
            num_workers=cfg.data.num_workers
        )

        model = QueryRefrenceImageEncoder(
            encoder=encoder,
            loss_fn=loss,
            optim_cfg=cfg.optim,
        )
    else:
        data = TrainerSingleImageSet(
            transform=transform, 
            dataset_name=cfg.data.name,
            batch_size=cfg.data.batch_size, 
            num_workers=cfg.data.num_workers,
            val_transform=torchvision.transforms.Normalize(cfg.data.mean, cfg.data.std)
        )

        model = ImageEncoder(
            encoder=encoder,
            loss_fn=loss,
            optim_cfg=cfg.optim,
            logits=cfg.encoder.logits
        )

    # set up wandb and trainers
    wandb.login(key=cfg.secrets.wandb_key)
    wandb_logger = WandbLogger(project='mqm', config=flatten_dict(cfg), entity='lambda-ai')

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints', 
        filename='{epoch}-{valid_loss:.2f}', 
        save_top_k=5, 
        monitor='valid_loss',
        save_weights_only=False
    )
    early_stop_callback = EarlyStopping(
        monitor="valid_loss", 
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min"
    )

    trainer = pl.Trainer(
        logger=wandb_logger,    
        log_every_n_steps=2,   
        gpus=None if not torch.cuda.is_available() else -1,
        max_epochs=500,           
        deterministic=True, 
        accumulate_grad_batches=cfg.n_batches_accumalation,
        precision=32 if not torch.cuda.is_available() else 16,   
        profiler="simple",
        gradient_clip_val=cfg.optim.gradient_clip_val,
        callbacks=[checkpoint_callback, early_stop_callback, RichModelSummary()],
    )

    # fit the model
    trainer.fit(model, data)
    wandb.finish()

if __name__ == "__main__":
    main()