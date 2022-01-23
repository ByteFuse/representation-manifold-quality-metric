from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import pandas as pd
import pytorch_lightning as pl

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.data.utils import (
    load_cub_dataset,
    load_cifar100_dataset,
    load_cifar10_dataset,
    load_cars_dataset,
    load_caltect_dataset,
    load_fashion_dataset,
    load_kmnist_dataset,
    load_mnist_dataset,
    load_omniglot_dataset
)
from src.models import CifarResNet18, LeNet

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


def quick_fine_tune(encoder, embedding_dim, dataloader, transform, n_classes=100):

    model = nn.Sequential(
        nn.ReLU(),
        nn.Linear(embedding_dim, n_classes)
    )

    params = list(encoder.parameters())+list(model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params)
    encoder.train()
    model.cuda()
    model.train()

    for epoch in range(25):
        running_loss = 0.0
        for batch in dataloader:
            images, labels = batch
            optimizer.zero_grad() 
            embeddings = encoder(transform(images).cuda())
            outputs = model(embeddings).cpu()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Loss: {round(running_loss, 3)} at epoch {epoch}')

    encoder.eval()
    model.cpu()

    return encoder

def main():

    DATASET = 'mnist'
    NEW_DATASET = 'fashion'
    FINE_TUNE = True
    N_CLASSES = 10

    loaders = {
        'cifar10': load_cifar10_dataset,
        'cifar100': load_cifar100_dataset,
        'cub': load_cub_dataset,
        'cars196': load_cars_dataset,
        'caltech': load_caltect_dataset,
        'fashion': load_fashion_dataset,
        'mnist': load_mnist_dataset,
        'omniglot': load_omniglot_dataset,
        'kmnist': load_kmnist_dataset,
    }

    train, test = loaders[NEW_DATASET]('./')

    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=512,
        shuffle=True,
        num_workers=5,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        test,
        batch_size=512,
        shuffle=False,
        num_workers=5,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False
    )

    
    if DATASET=='cifar10':
        transform = torchvision.transforms.Normalize([0.49139968, 0.48215827 ,0.44653124],[0.24703233, 0.24348505, 0.26158768])
    else:
       transform = torchvision.transforms.Normalize((0.1307,), (0.3081,))


    embedding_dims = [3,16,32,64,128,256,512]

    results = pd.DataFrame()
    
    for optim in ['adam','sgd']:
        for method in ['cross-entropy', 'triplet-supervised', 'triplet', 'triplet-entropy', 'random', 'nt-xent']:
            for embedding_dim in embedding_dims:
                
                
                print(f'Starting calculation for method: {method} and embedding dim: {embedding_dim}')

                if DATASET=='cifar10':
                    cnn_encoder = CifarResNet18(
                        embedding_dim=embedding_dim, 
                        hidden_dim=1024,
                        logits=True if method in ['cross-entropy', 'triplet-entropy'] else False,
                        number_classes=10 if method in ['cross-entropy', 'triplet-entropy'] else None,
                    )
                else:
                    cnn_encoder = LeNet(
                        embedding_dim=embedding_dim, 
                        logits=True if method in ['cross-entropy', 'triplet-entropy'] else False,
                        number_classes=10 if method in ['cross-entropy', 'triplet-entropy'] else None,
                    )

                model = GenericPlaceHolder(encoder=cnn_encoder)

                if method!='random':
                    encoder_checkpoint = f"./multirun/data={DATASET}/{method}/{optim}/encoder.embedding_dim={embedding_dim}/checkpoints/last.ckpt"
                    model = model.load_from_checkpoint(encoder_checkpoint)
                    print(f"Loaded encoder from {encoder_checkpoint}")	
                encoder = model.encoder
                encoder.logits=False
                encoder.eval()
                encoder.cuda()

                if FINE_TUNE:
                    print('Fine tuning encoder for 25 epochs')
                    encoder = quick_fine_tune(encoder, embedding_dim, train_dataloader, transform, n_classes=N_CLASSES)


                train_embeddings = np.zeros(shape=(1, embedding_dim))
                labels = []
                for batch in tqdm(train_dataloader, desc='Training data generation', leave=True):
                    images, labs = batch
                    labels.extend([int(l) for l in labs])
                    images = transform(images)

                    with torch.no_grad():
                        reps = encoder(images.cuda()).cpu().numpy()
                    train_embeddings = np.concatenate((train_embeddings, reps))
                train_embeddings = train_embeddings[1:]

                test_embeddings = np.zeros(shape=(1, embedding_dim))
                test_labels = []
                for batch in tqdm(test_dataloader, desc='Test data generation', leave=True):
                    images, labs = batch
                    test_labels.extend([int(l) for l in labs])
                    images = transform(images)

                    with torch.no_grad():
                        reps = encoder(images.cuda()).cpu().numpy()
                    test_embeddings = np.concatenate((test_embeddings, reps))
                test_embeddings = test_embeddings[1:]

                train_embeddings = F.normalize(torch.tensor(train_embeddings), p=2, dim=1).numpy()
                test_embeddings = F.normalize(torch.tensor(test_embeddings), p=2, dim=1).numpy()


                print('Finding best KNN')
                knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=1)
                knn.fit(train_embeddings, labels)

                print('Getting y hat')
                y_hat = knn.predict(test_embeddings)
                accuracy = accuracy_score(test_labels, y_hat)
                print(accuracy)

                _ = pd.DataFrame({'model':[method], 'optim':[optim], 'embedding_dim':[embedding_dim], 'accuracy':[accuracy]})
                results = pd.concat([results, _])
                
                if FINE_TUNE:
                    results.to_csv(f'F://results/search/knn_results_{DATASET}_to_{NEW_DATASET}_finetuned.csv', index=False)
                else:
                    results.to_csv(f'F://results/search/knn_results_{DATASET}_to_{NEW_DATASET}.csv', index=False)
                print('*'*50)
if __name__ == "__main__":
    main()