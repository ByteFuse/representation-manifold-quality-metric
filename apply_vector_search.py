from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

import pandas as pd
import pytorch_lightning as pl

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

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


def main():

    DATASET = 'mnist'
    NEW_DATASET = 'fashion'

    loaders = {
        'cifar10': load_cifar10_dataset,
        'cifar100': load_cifar100_dataset,
        'cub': load_cub_dataset,
        'cars196': load_cars_dataset,
        'caltech': load_caltect_dataset,
        'fashion': load_fashion_dataset,
        'mnist': load_mnist_dataset,
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
    
    for optim in ['adam', 'sgd']:
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

                train_embeddings = np.zeros(shape=(1, embedding_dim))
                labels = []
                encoder.cuda()
                for batch in tqdm(train_dataloader, desc='Training data generation', leave=False):
                    images, labs = batch
                    labels.extend([int(l) for l in labs])
                    images = transform(images)

                    with torch.no_grad():
                        reps = encoder(images.cuda()).cpu().numpy()
                    train_embeddings = np.concatenate((train_embeddings, reps))
                train_embeddings = train_embeddings[1:]

                test_embeddings = np.zeros(shape=(1, embedding_dim))
                test_labels = []
                encoder.cuda()
                for batch in tqdm(test_dataloader, desc='Test data generation', leave=False):
                    images, labs = batch
                    test_labels.extend([int(l) for l in labs])
                    images = transform(images)

                    with torch.no_grad():
                        reps = encoder(images.cuda()).cpu().numpy()
                    test_embeddings = np.concatenate((test_embeddings, reps))
                test_embeddings = test_embeddings[1:]

                train_embeddings = F.normalize(torch.tensor(train_embeddings), p=2, dim=1).numpy()
                test_embeddings = F.normalize(torch.tensor(test_embeddings), p=2, dim=1).numpy()


                parameters_KNN = {
                    'n_neighbors': (1,5, 10,30),
                }
                knn = KNeighborsClassifier(n_jobs=4)
                grid_search_KNN = GridSearchCV(
                    estimator=knn,
                    param_grid=parameters_KNN,
                    scoring = 'accuracy',
                    n_jobs = 4,
                    cv = 3
                )

                best_knn = grid_search_KNN.fit(train_embeddings, labels)
                print(f'Best parameters for KNN: {best_knn.best_params_}')
                
                y_hat = grid_search_KNN.predict(test_embeddings)
                accuracy = accuracy_score(test_labels, y_hat)
                print(accuracy)

                _ = pd.DataFrame({'model':method, 'optim':optim, 'embedding_dim':embedding_dim, 'accuracy':accuracy})
                results = pd.concat([results, _])

                results.to_csv('knn_results_{DATASET}_to_{NEW_DATASET}.csv', index=False)

if __name__ == "__main__":
    main()