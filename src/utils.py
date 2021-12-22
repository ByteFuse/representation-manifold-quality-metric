from collections.abc import MutableMapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from umap import UMAP

import torch
import torchvision


def return_fgsm_contrastive_attack_images(images, model, loss_fn, val_transform, transform, device='cuda', epsilon=0):

    images = images.to(device)
    
    ref_images = transform(images)
    query_images = val_transform(images)
    query_images.requires_grad = True

    ref_emb, query_emb = model(ref_images), model(query_images)
    model.zero_grad()
    loss = loss_fn(ref_emb, query_emb).to(device)
    loss.backward()
    attack_images = torch.clip(images + epsilon*query_images.grad.data.sign(), 0, 1)
    
    return attack_images

def return_fgsm_supervised_attack_images(images, labels, model, loss_fn, final_transform, require_logits=False, device='cuda', epsilon=0):
    
    images_return = images.clone().to(device)
    
    images = final_transform(images.to(device))
    labels = labels.to(device)
    images.requires_grad = True

    if require_logits:
        embeddings, logits = model(images)
        model.zero_grad()
        loss = loss_fn(embeddings, logits, labels).to(device)
    else:
        embeddings = model(images)
        model.zero_grad()
        loss = loss_fn(embeddings, labels).to(device)

    loss.backward()
    attack_images = torch.clip(images_return + epsilon*images.grad.data.sign(), 0, 1)
    
    return attack_images

def plot_embeddings_unimodal(plot_data, epoch, return_fig=False):
    embeddings = []
    labels = []

    for data in plot_data:
        embedding, label = data[0], data[1]
        labels.extend(label)

        for emb in embedding:
            embeddings.append(emb.detach().cpu().numpy())

    labels = [str(int(i.cpu())) for i in labels]

    embeddings = UMAP().fit_transform(embeddings)
    embeddings = np.array(embeddings)

    df = pd.DataFrame({
        'comp1': embeddings[:,0],
        'comp2': embeddings[:,1],
        'labels': labels,
    })

    df = df.sort_values('labels')

    fig = plt.figure(figsize=(15,10))
    sns.scatterplot(x=df.comp1, y=df.comp2, hue=df.labels)
    plt.legend()
    plt.title(f'Projected validation embeddings at epoch: {epoch}')

    if return_fig:
        return fig
    else:
        plt.show()


def plot_batch_of_images(batch, return_fig=False):
    fig = plt.figure(figsize=(15,20))
    plt.imshow(torchvision.utils.make_grid(batch, nrow=batch.size(0)//6).permute(1,2,0))

    if return_fig:
        return fig
    else:
        plt.show()


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    return dict(_flatten_dict_gen(d, parent_key, sep))
