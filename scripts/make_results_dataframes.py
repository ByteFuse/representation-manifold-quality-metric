import os
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
from src.metrics import return_distances, return_distances_label_wise

import warnings
warnings.filterwarnings('ignore')


DATASET='cifar10'
ADVERSERIAL = True
p=2

if DATASET=='cifar10':
    classes = {0:'airplane',
               1:'automobile',
               2:'bird',
               3:'cat',
               4:'deer',
               5:'dog',
               6:'frog',
               7:'horse',
               8:'ship',
               9:'truck',}
else:
    classes = {0:'Digit=0',
               1:'Digit=1',
               2:'Digit=2',
               3:'Digit=3',
               4:'Digit=4',
               5:'Digit=5',
               6:'Digit=6',
               7:'Digit=7',
               8:'Digit=8',
               9:'Digit=9',}
    
    
models = [
    f'xent_{DATASET}',
    f'tripent_{DATASET}',
    f'trip_sup_{DATASET}',
    f'ntxent_{DATASET}',
    f'trip_{DATASET}',
    f'random_init'
]

models_nice_name = [
    'Cross-Entropy',
    'Triplet-Entropy',
    'Triplet-Supervised',
    'NT-XENT',
    'Triplet-Loss',
    'Random'
]

cols = ['#003f5c', '#5ab81c', '#b853ae', '#b83014', '#3f94b8']

colors = {m:c for m, c in zip(models,cols)}

nice_names = {m:mn for m, mn in zip(models,models_nice_name)}

epsilons = np.linspace(0,1,100)
pgds = list(range(30))

if ADVERSERIAL:
    meta_data = ['pgd_iterations', 'model', 'image_index', 'label']
    alterations='pgd'
    n=30
else:
    meta_data = ['epsilon', 'model', 'image_index', 'label']
    alterations='noise'
    n=100


high_level = pd.DataFrame()
plot = pd.DataFrame()

high_level_label = pd.DataFrame()
plot_label = pd.DataFrame()

for OPTIM in tqdm(['adam', 'sgd'], desc='OPTIM'):
    for embedding_dim in tqdm([128], desc='embedding'):

        if ADVERSERIAL:
            path = f'../results/data={DATASET}/{OPTIM}/ADVERSERIAL_attacks/embedding_dim={embedding_dim}/'
        else:
            path = f'../results/data={DATASET}/{OPTIM}/embedding_dim={embedding_dim}/'

        df = pd.DataFrame()
        for f in tqdm(os.listdir(path), desc='loading'):
            try:
                if ADVERSERIAL:
                    _ = pd.read_pickle(os.path.join(path,f))
                    df = pd.concat([df,_])
                else:
                    if 'run0' in f:
                        _ = pd.read_pickle(os.path.join(path,f))
                        df = pd.concat([df,_])
            except:
                pass
            

        plot_label_based, label_based = return_distances_label_wise(
            df,
            models,
            10, 
            n, 
            meta_data,
            p,
            embedding_dim, 
            False
        )
        plot_points_based, points_based = return_distances(
            df,
            models,
            n,
            meta_data,
            p,
            embedding_dim,
            False
        )

        del df
        
        plot_label_based['embedding_dim'] = embedding_dim
        label_based['embedding_dim'] = embedding_dim
        plot_points_based['embedding_dim'] = embedding_dim
        points_based['embedding_dim'] = embedding_dim
        
        plot_label_based['optim'] = OPTIM
        label_based['optim'] = OPTIM
        plot_points_based['optim'] = OPTIM
        points_based['optim'] = OPTIM
        
        plot_label_based['alteration_type'] = alterations
        label_based['alteration_type'] = alterations
        plot_points_based['alteration_type'] = alterations
        points_based['alteration_type'] = alterations
        
                
        plot_label_based['p'] = p
        label_based['p'] = p
        plot_points_based['p'] = p
        points_based['p'] = p
        
        high_level = pd.concat([high_level, points_based])
        plot = pd.concat([plot, plot_points_based])
        high_level_label = pd.concat([high_level_label, label_based])
        plot_label = pd.concat([plot_label, plot_label_based ])
        
        high_level.to_csv(f'../results/distances/results_{DATASET}_{alterations}_p-{p}_high_level_data.csv', index=False)
        plot.to_csv(f'../results/distances/results_{DATASET}_{alterations}_p-{p}_plot_data.csv', index=False)
        high_level_label.to_csv(f'../results/distances/results_{DATASET}_{alterations}_p-{p}_high_level_label_data.csv', index=False)
        plot_label.to_csv(f'../results/distances/results_{DATASET}_{alterations}_p-{p}_plot_label_data.csv', index=False)