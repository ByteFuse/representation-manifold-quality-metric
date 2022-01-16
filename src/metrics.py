from tqdm import tqdm

import pandas as pd

import numpy as np
import torch 
import torch.nn.functional as F


def calculate_pointwise_distances_metrics(points, number_of_alterations, p, embedding_dim):

    points = F.normalize(torch.tensor(points), p=2, dim=1)   
    points = points.reshape(points.shape[0]//number_of_alterations, -1 , embedding_dim) # (number_of_alterations*N, embedding_dim) -> (N, number_of_alterations, embedding_dim)

    if points.shape[-1]>512:
        distances = []
        for i, j in zip(range(0, points.shape[0], 1000), range(1000, points.shape[0]+1000, 1000)):
            distances.append(torch.cdist(points[i:j], points[i:j], p=p))
        distance_matrix = torch.cat(distances, 0)
    else:
        distance_matrix = torch.cdist(points, points, p=p)

    distance_towards_original = distance_matrix[:, 1:, 0] # (N, number_of_alterations) <- row is distances to original point
    distance_towards_previous = torch.diagonal(distance_matrix, offset=-1, dim1=1, dim2=2)

    df_original = pd.DataFrame(distance_towards_original.numpy())
    df_previous = pd.DataFrame(distance_towards_previous.numpy())

    pct_changes_original = df_original.pct_change(axis=1).drop(0,axis=1) # we know first value will be nan
    pct_changes_previous = df_previous.pct_change(axis=1).drop(0,axis=1)

    # filter out points that mess stuff up
    pct_changes_original.replace([np.inf, -np.inf], np.nan, inplace=True)
    pct_changes_previous.replace([np.inf, -np.inf], np.nan, inplace=True)
    pct_changes_original = pct_changes_original.dropna().values
    pct_changes_previous = pct_changes_previous.dropna().values

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
    plot_results = pd.DataFrame()
    
    for model in tqdm(models_to_investigate):
        for label in range(labels_range):

            points = dataframe[(dataframe.model==model)&(dataframe.label==label)].\
                            sort_values(['image_index', meta_data_collumns[0]]).\
                            drop(meta_data_collumns, axis=1).values

            if centroid:
                distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous = calculate_centroid_distances_metrics(
                    points, number_of_alterations, p, embedding_dim)
                                
                #TODO: add the centroid dataframes
                
                
            else:
                distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous = calculate_pointwise_distances_metrics(
                    points, number_of_alterations, p, embedding_dim)

                total_spikes_per_point = np.abs(pct_changes_original).sum(axis=1)
                total_spikes_per_point_previous = np.abs(pct_changes_previous).sum(axis=1)

                average_spike_per_point = np.abs(pct_changes_original).mean(axis=1)
                average_spike_per_point_previous = np.abs(pct_changes_previous).mean(axis=1)

                total_distance_walked_per_point = distance_towards_previous.sum(axis=1)
                total_distances_moved_per_point = distance_towards_original.sum(axis=1)

                average_distance_increase_per_point = distance_towards_original.mean(axis=1)
                average_distance_increase_per_point_previous= distance_towards_previous.mean(axis=1)

                total_spikes_average = total_spikes_per_point.mean()
                total_spikes_std_error = total_spikes_per_point.std()
                total_spikes_average_previous = total_spikes_per_point_previous.mean()
                total_spikes_std_error_previous = total_spikes_per_point_previous.std()
                total_spikes = total_spikes_per_point.sum()
                total_spikes_revious = total_spikes_per_point_previous.sum()

                average_spike = average_spike_per_point.mean()
                average_spike_std_error = average_spike_per_point.std()
                average_spike_previous = average_spike_per_point_previous.mean()
                average_spike_previous_std_error = average_spike_per_point_previous.mean()

                total_walking_distance_average = total_distance_walked_per_point.mean()
                total_walking_distance_std_error = total_distance_walked_per_point.std()
                total_distance_moved_average = total_distances_moved_per_point.mean()
                total_distance_moved_std_error = total_distances_moved_per_point.std()

                average_distance_increase = average_distance_increase_per_point.mean()
                average_spike_previous_std_error = average_distance_increase_per_point.std()
                average_distance_increase_previous = average_distance_increase_per_point_previous.mean()
                average_distance_increase_previous_std_error = average_distance_increase_per_point_previous.mean()

                plot_data = pd.DataFrame({
                    'original_pct_change_mean':np.abs(pct_changes_original).mean(0),
                    'original_pct_change_std_error':np.abs(pct_changes_original).std(0),
                    'original_mean':distance_towards_original[:,1:].mean(0),
                    'original_std_error':distance_towards_original[:,1:].std(0),
                    'previous_pct_change_mean':np.abs(pct_changes_previous).mean(0),
                    'previous_pct_change_std_error':np.abs(pct_changes_previous).std(0),
                    'previous_mean':distance_towards_previous[:,1:].mean(0),
                    'previous_std_error':distance_towards_previous[:,1:].std(0),
                    'model':[model]*(number_of_alterations-2),
                    'label':[label]*(number_of_alterations-2),
                    'alteration':list(range(2,number_of_alterations))
                })

                high_level_data = pd.DataFrame({
                    'total_spikes_average':[float(total_spikes_average)],
                    'total_spikes_std_error':[float(total_spikes_std_error)],
                    'total_spikes_average_previous':[float(total_spikes_average_previous)],
                    'total_spikes_std_error_previous':[float(total_spikes_std_error_previous)],
                    'total_spikes':[float(total_spikes)],
                    'total_spikes_revious':[float(total_spikes_revious)],
                    'average_spike':[float(average_spike)],
                    'average_spike_std_error':[float(average_spike_std_error)],
                    'average_spike_previous':[float(average_spike_previous)],
                    'average_spike_previous_std_error':[float(average_spike_previous_std_error)],
                    'total_walking_distance_average':[float(total_walking_distance_average)],
                    'total_walking_distance_std_error':[float(total_walking_distance_std_error)],
                    'total_distance_moved_average':[float(total_distance_moved_average)],
                    'total_distance_moved_std_error':[float(total_distance_moved_std_error)],
                    'average_distance_increase':[float(average_distance_increase)],
                    'average_spike_previous_std_error':[float(average_spike_previous_std_error)],
                    'average_distance_increase_previous':[float(average_distance_increase_previous)],
                    'average_distance_increase_previous_std_error':[float(average_distance_increase_previous_std_error)],
                    'label':[label],
                    'model':[model],
                })
                
            plot_results = pd.concat([plot_results, plot_data])
            high_level_results = pd.concat([high_level_results, high_level_data])

    return plot_results, high_level_results


def return_distances(
    dataframe,
    models_to_investigate,
    number_of_alterations,
    meta_data_collumns,
    p,
    embedding_dim,
    centroid=False):
    
    high_level_results = pd.DataFrame()
    plot_results = pd.DataFrame()

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
            
            #TODO: add the centroid dataframes
            
        else:
            distance_towards_original, distance_towards_previous, pct_changes_original, pct_changes_previous = calculate_pointwise_distances_metrics(
                points,
                number_of_alterations, 
                p, 
                embedding_dim)

            total_spikes_per_point = np.abs(pct_changes_original).sum(axis=1)
            total_spikes_per_point_previous = np.abs(pct_changes_previous).sum(axis=1)

            average_spike_per_point = np.abs(pct_changes_original).mean(axis=1)
            average_spike_per_point_previous = np.abs(pct_changes_previous).mean(axis=1)

            total_distance_walked_per_point = distance_towards_previous.sum(axis=1)
            total_distances_moved_per_point = distance_towards_original.sum(axis=1)

            average_distance_increase_per_point = distance_towards_original.mean(axis=1)
            average_distance_increase_per_point_previous= distance_towards_previous.mean(axis=1)

            total_spikes_average = total_spikes_per_point.mean()
            total_spikes_std_error = total_spikes_per_point.std()
            total_spikes_average_previous = total_spikes_per_point_previous.mean()
            total_spikes_std_error_previous = total_spikes_per_point_previous.std()
            total_spikes = total_spikes_per_point.sum()
            total_spikes_revious = total_spikes_per_point_previous.sum()

            average_spike = average_spike_per_point.mean()
            average_spike_std_error = average_spike_per_point.std()
            average_spike_previous = average_spike_per_point_previous.mean()
            average_spike_previous_std_error = average_spike_per_point_previous.mean()

            total_walking_distance_average = total_distance_walked_per_point.mean()
            total_walking_distance_std_error = total_distance_walked_per_point.std()
            total_distance_moved_average = total_distances_moved_per_point.mean()
            total_distance_moved_std_error = total_distances_moved_per_point.std()

            average_distance_increase = average_distance_increase_per_point.mean()
            average_spike_previous_std_error = average_distance_increase_per_point.std()
            average_distance_increase_previous = average_distance_increase_per_point_previous.mean()
            average_distance_increase_previous_std_error = average_distance_increase_per_point_previous.mean()

            plot_data = pd.DataFrame({
                'original_pct_change_mean':np.abs(pct_changes_original).mean(0),
                'original_pct_change_std_error':np.abs(pct_changes_original).std(0),
                'original_mean':distance_towards_original[:,1:].mean(0),
                'original_std_error':distance_towards_original[:,1:].std(0),
                'previous_pct_change_mean':np.abs(pct_changes_previous).mean(0),
                'previous_pct_change_std_error':np.abs(pct_changes_previous).std(0),
                'previous_mean':distance_towards_previous[:,1:].mean(0),
                'previous_std_error':distance_towards_previous[:,1:].std(0),
                'model':[model]*(number_of_alterations-2),
                'alteration':list(range(2,number_of_alterations))
            })

            high_level_data = pd.DataFrame({
                'total_spikes_average':[float(total_spikes_average)],
                'total_spikes_std_error':[float(total_spikes_std_error)],
                'total_spikes_average_previous':[float(total_spikes_average_previous)],
                'total_spikes_std_error_previous':[float(total_spikes_std_error_previous)],
                'total_spikes':[float(total_spikes)],
                'total_spikes_revious':[float(total_spikes_revious)],
                'average_spike':[float(average_spike)],
                'average_spike_std_error':[float(average_spike_std_error)],
                'average_spike_previous':[float(average_spike_previous)],
                'average_spike_previous_std_error':[float(average_spike_previous_std_error)],
                'total_walking_distance_average':[float(total_walking_distance_average)],
                'total_walking_distance_std_error':[float(total_walking_distance_std_error)],
                'total_distance_moved_average':[float(total_distance_moved_average)],
                'total_distance_moved_std_error':[float(total_distance_moved_std_error)],
                'average_distance_increase':[float(average_distance_increase)],
                'average_spike_previous_std_error':[float(average_spike_previous_std_error)],
                'average_distance_increase_previous':[float(average_distance_increase_previous)],
                'average_distance_increase_previous_std_error':[float(average_distance_increase_previous_std_error)],
                'model':[model],
            })
        
        plot_results = pd.concat([plot_results, plot_data])
        high_level_results = pd.concat([high_level_results, high_level_data])

    return plot_results, high_level_results



