from tqdm import tqdm

import pandas as pd

import numpy as np
import torch 
import torch.nn.functional as F


def calculate_pointwise_distances_metrics(points, number_of_alterations, p, embedding_dim):

    points = F.normalize(torch.tensor(points), p=2, dim=1)   
    points = points.reshape(points.shape[0]//number_of_alterations, -1, embedding_dim) # (number_of_alterations*N, embedding_dim) -> (N, number_of_alterations, embedding_dim)

    distance_matrix = torch.cdist(points, points, p=p)
    distance_towards_original = distance_matrix[:, 1:, 0]
    distance_towards_previous = torch.diagonal(distance_matrix, offset=-1, dim1=1, dim2=2)

    df_original = pd.DataFrame(distance_towards_original.numpy())
    df_previous = pd.DataFrame(distance_towards_previous.numpy())

    pct_changes_original = df_original.pct_change(axis=1).values
    pct_changes_previous = df_previous.pct_change(axis=1).values

    return distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous


def calculate_centroid_distances_metrics(points, number_of_alterations, p, embedding_dim):

    points = F.normalize(torch.tensor(points), p=2, dim=1)   
    points = points.reshape(number_of_alterations, -1, embedding_dim) # (number_of_alterations*N, embedding_dim) -> (number_of_alterations, N, embedding_dim)

    centriods = torch.mean(points, dim=1)
    distance_matrix = torch.cdist(centriods, centriods, p=p)

    distance_towards_original = distance_matrix[1:, 0]
    distance_towards_previous = torch.diagonal(distance_matrix, offset=-1)

    df_original = pd.DataFrame(distance_towards_original.numpy())
    df_previous = pd.DataFrame(distance_towards_previous.numpy())

    pct_changes_original = df_original.pct_change().values.T[0]
    pct_changes_previous = df_previous.pct_change().values.T[0]

    return distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous



def return_distances_label_wise(
    dataframe,
    models_to_investigate,
    labels_range,
    number_of_alterations,
    meta_data_collumns,
    p,
    embedding_dim,
    centroid=False):

    high_level_results = pd.DataFrame()
    
    for model in tqdm(models_to_investigate):
        for label in range(labels_range):

            points = dataframe[(dataframe.model==model)&(dataframe.label==label)].\
                            sort_values(['image_index', meta_data_collumns[0]]).\
                            drop(meta_data_collumns, axis=1).values

            if centroid:
                distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous = calculate_centroid_distances_metrics(
                    points, number_of_alterations, p, embedding_dim)
                                
                _ = pd.DataFrame({
                    'original_pct_change_mean':np.abs(pct_changes_original),
                    'original_mean':distance_towards_original.numpy(),
                    'previous_pct_change_mean':np.abs(pct_changes_previous),
                    'previous_mean':distance_towards_previous.numpy(),
                    'label':[label]*(number_of_alterations-1),
                    'model':[model]*(number_of_alterations-1),
                    'alteration':list(range(number_of_alterations-1))
                })
                
                
            else:
                distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous = calculate_pointwise_distances_metrics(
                    points, number_of_alterations, p, embedding_dim)
                
            
                _ = pd.DataFrame({
                    'original_pct_change_mean':np.abs(pct_changes_original).mean(0),
                    'original_pct_change_std':np.abs(pct_changes_original).std(0),
                    'original_mean':distance_towards_original.mean(0),
                    'original_std':distance_towards_original.std(0), 
                    'previous_pct_change_mean':np.abs(pct_changes_previous).mean(0),
                    'previous_pct_change_std':np.abs(pct_changes_previous).std(0),
                    'previous_mean':distance_towards_previous.mean(0),
                    'previous_std':distance_towards_previous.std(0), 
                    'label':[label]*(number_of_alterations-1),
                    'model':[model]*(number_of_alterations-1),
                    'alteration':list(range(number_of_alterations-1))
                })
            
            high_level_results = pd.concat([high_level_results, _[1:]])

    return high_level_results


def return_distances(
    dataframe,
    models_to_investigate,
    number_of_alterations,
    meta_data_collumns,
    p,
    embedding_dim,
    centroid=False):
    
    high_level_results = pd.DataFrame()

    for model in tqdm(models_to_investigate):
        points = dataframe[dataframe.model==model].\
                        sort_values(['image_index', meta_data_collumns[0]]).\
                        drop(meta_data_collumns, axis=1).values

        if centroid:
            distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous = calculate_centroid_distances_metrics(
                points, 
                number_of_alterations, 
                p, 
                embedding_dim)
            
            _ = pd.DataFrame({
                    'original_pct_change_mean':np.abs(pct_changes_original),
                    'original_mean':distance_towards_original.numpy(),
                    'previous_pct_change_mean':np.abs(pct_changes_previous),
                    'previous_mean':distance_towards_previous.numpy(),
                    'model':[model]*(number_of_alterations-1),
                    'alteration':list(range(number_of_alterations-1))
                })
        else:
            distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous = calculate_pointwise_distances_metrics(
                points,
                number_of_alterations, 
                p, 
                embedding_dim)

            _ = pd.DataFrame({
                'original_pct_change_mean':np.abs(pct_changes_original).mean(0),
                'original_pct_change_std':np.abs(pct_changes_original).std(0),
                'original_mean':distance_towards_original.mean(0),
                'original_std':distance_towards_original.std(0), 
                'previous_pct_change_mean':np.abs(pct_changes_previous).mean(0),
                'previous_pct_change_std':np.abs(pct_changes_previous).std(0),
                'previous_mean':distance_towards_previous.mean(0),
                'previous_std':distance_towards_previous.std(0), 
                'model':[model]*(number_of_alterations-1),
                'alteration':list(range(number_of_alterations-1))})

        high_level_results = pd.concat([high_level_results, _[1:]])

    return high_level_results



