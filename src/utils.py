from collections.abc import MutableMapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from umap import UMAP

import torchvision


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
