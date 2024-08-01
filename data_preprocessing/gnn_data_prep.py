# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:09:58 2024

@author: cnmlt
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import random

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def generate_adjacency_matrix(coordinates, threshold=0.5):
    """
    Generate adjacency matrix based on spatial coordinates and a given threshold.
    """
    num_nodes = len(coordinates)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    dist_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = euclidean_distance(coordinates[i, 0], coordinates[i, 1], coordinates[j, 0], coordinates[j, 1])
            dist_matrix[i, j] = distance
            if distance < threshold:
                adj_matrix[i, j] = adj_matrix[j, i] = 1

    return adj_matrix


def generate_complex_adjacency_matrix(all_coords, num_piezo, num_pump, num_prec, num_evap, num_river, percentage=None, n_piezo_connected = 3):
    """
    Generate a complex adjacency matrix based on spatial coordinates and specific connectivity rules.
    
    Parameters:
    - all_coords: Numpy array of coordinates for all points (piezometers, pumps, precipitation, evaporation, river points).
    - num_piezo, num_pump, num_prec, num_evap, num_river: Number of each type of point.
    """
    num_nodes = len(all_coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    dist_matrix = np.zeros((num_nodes, num_nodes))

    # Compute the distance matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i, j] = euclidean_distance(all_coords[i, 0], all_coords[i, 1], all_coords[j, 0], all_coords[j, 1])

    # Always connect each piezometer to the 3 closest piezometers
    for i in range(num_piezo):
            # Connect to the 3 closest piezometers (excluding itself)
            piezo_indices = np.argsort(dist_matrix[i, :num_piezo])[2:2+n_piezo_connected]  # Skip the first index (itself)
            adj_matrix[i, piezo_indices] = 0.1

    # Determine the indices of nodes to connect to exhogenous variables
    if percentage is not None:
        # Calculate the number of nodes to process based on the percentage
        num_nodes_to_process = int(np.ceil(num_nodes * (percentage / 100.0)))
        # Randomly select the indices of the nodes to process
        selected_indices = random.sample(range(num_nodes), num_nodes_to_process)
    else:
        # If no percentage is given, process all nodes
        
        selected_indices = range(num_nodes)


    # Connectivity logic
    for i in selected_indices:
        if i < num_piezo:  # For piezometers
            # Connect to the 3 closest piezometers (excluding itself)
            # piezo_indices = np.argsort(dist_matrix[i, :num_piezo])[2:5]  # Skip the first index (itself)
            # adj_matrix[i, piezo_indices] = 0.1

            # Connect to all pumps
            # pump_indices = range(num_piezo, num_piezo + num_pump)
            pump_indices = num_piezo + np.argmin(dist_matrix[i, num_piezo : num_piezo + num_pump])

            adj_matrix[i, pump_indices] = 0.2

            # Connect to the closest precipitation
            prec_index = num_piezo + num_pump + np.argmin(dist_matrix[i, num_piezo + num_pump:num_piezo + num_pump + num_prec])
            adj_matrix[i, prec_index] = 0.3

            # Connect to the closest evaporation
            evap_index = num_piezo + num_pump + num_prec + np.argmin(dist_matrix[i, num_piezo + num_pump + num_prec:num_piezo + num_pump + num_prec + num_evap])
            adj_matrix[i, evap_index] = 0.4

            # Connect to the two closest rivers
            river_indices = np.argsort(dist_matrix[i, -num_river:])[:2] + (num_nodes - num_river)
            adj_matrix[i, river_indices] = 0.5

    # Symmetrize the matrix for undirected connections
    adj_matrix = adj_matrix + adj_matrix.T

    return adj_matrix





def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def create_static_features(all_x, all_y, all_z, all_type):
    """
    Prepare static features for each node, such as spatial coordinates and categorical types.
    """
    # Normalize features
    x_normalized = normalize(all_x)
    y_normalized = normalize(all_y)
    z_normalized = normalize(all_z)

    # Combine into a single tensor
    static_features = torch.tensor(np.column_stack((x_normalized, y_normalized, z_normalized, all_type)), dtype=torch.float)

    return static_features

import os

def load_and_concatenate_metadata(piezo_metadata_path, pump_metadata_path, evap_metadata_path, prec_metadata_path, river_metadata_path, df_piezo_columns, pump_columns, locations_no_missing):
    # Load metadata for piezometers
    piezo_metadata = pd.read_csv(piezo_metadata_path)
    filtered_metadata = piezo_metadata[piezo_metadata['name'].isin(df_piezo_columns)]
    filtered_metadata = filtered_metadata.set_index('name').reindex(df_piezo_columns).reset_index()
    
    # Load pump locations
    pump_metadata = pd.read_csv(pump_metadata_path)
    pump_metadata = pump_metadata[pump_metadata['Naam'].isin(pump_columns)]
    pump_metadata = pump_metadata.set_index('Naam').reindex(pump_columns).reset_index()
    
    # Load precipitation and evaporation locations
    evap_metadata = pd.read_csv(evap_metadata_path)
    prec_metadata = pd.read_csv(prec_metadata_path)
    
    # Load river locations
    river_metadata = pd.read_csv(river_metadata_path)
    river_metadata = river_metadata[river_metadata['name'].isin(locations_no_missing)]
    
    # Concatenate all locations and counts
    return (
        np.concatenate([filtered_metadata.iloc[:, 3].to_numpy(), pump_metadata['Xcoor'].to_numpy(), prec_metadata['x'].to_numpy(), evap_metadata['x'].to_numpy(), river_metadata['x'].to_numpy()]),
        np.concatenate([filtered_metadata.iloc[:, 4].to_numpy(), pump_metadata['Ycoor'].to_numpy(), prec_metadata['y'].to_numpy(), evap_metadata['y'].to_numpy(), river_metadata['y'].to_numpy()]),
        np.concatenate([filtered_metadata.iloc[:, 5].to_numpy(), np.zeros_like(pump_metadata['Xcoor'].to_numpy()), np.zeros_like(prec_metadata['x'].to_numpy()), np.zeros_like(evap_metadata['x'].to_numpy()), np.zeros_like(river_metadata['x'].to_numpy())]),
        np.concatenate([np.ones_like(filtered_metadata.iloc[:, 3].to_numpy()), 2*np.ones_like(pump_metadata['Xcoor'].to_numpy()), 3*np.ones_like(prec_metadata['x'].to_numpy()), 4*np.ones_like(evap_metadata['x'].to_numpy()), 5*np.ones_like(river_metadata['x'].to_numpy())]),
        len(filtered_metadata.iloc[:, 3]),  # num_piezo
        len(pump_metadata['Xcoor']),  # num_pump
        len(prec_metadata['x']),  # num_prec
        len(evap_metadata['x']),  # num_evap
        len(river_metadata['x'])  # num_river
    )

   

def main(df_piezo_columns, pump_columns, locations_no_missing, percentage=None, n_piezo_connected=3):
    # Paths to the metadata files (update these paths according to your folder structure)

    metadata_path = "C:/codes/wells_time/scripts/data/input/piezometers/oseries_metadata_and_selection.csv"
    pump_metadata_path = "C:/codes/wells_time/scripts/data/input/wells/productielocaties_wwt.csv"
    evap_metadata_path = "C:/codes/wells_time/scripts/data/input/meteo_metadata_and_timeseries/evap_metadata.csv"
    prec_metadata_path = "C:/codes/wells_time/scripts/data/input/meteo_metadata_and_timeseries/prec_metadata.csv"
    river_metadata_path = "C:/codes/wells_time/scripts/data/input/river/rivers_metadata.csv"
    
    all_x, all_y, all_z, all_type, num_piezo, num_pump, num_prec, num_evap, num_river = load_and_concatenate_metadata(
        metadata_path, pump_metadata_path, evap_metadata_path, prec_metadata_path, river_metadata_path,
        df_piezo_columns, pump_columns, locations_no_missing
    )
    
    # Prepare static features
    static_features = create_static_features(all_x, all_y, all_z, all_type)
    
    # Generate adjacency matrix
    # For this, you need to adjust coordinates format and threshold as needed
    coordinates = np.stack((all_x, all_y), axis=-1)
    # adj_matrix = generate_adjacency_matrix(coordinates, threshold=0.5)  # Adjust threshold as needed
    adj_matrix = generate_complex_adjacency_matrix(coordinates, num_piezo, num_pump, num_prec, num_evap, num_river, percentage, n_piezo_connected)
    
    base_data_path = Path('C:/codes/wells_time/scripts/data/preprocessed/').absolute()

    # Save adj_matrix and static_features for later use in your GNN model
    # torch.save(adj_matrix, base_data_path/ 'adj_matrix.pt')
    # torch.save(static_features, base_data_path / 'static_features.pt')

    adj_matrix_tensor = torch.tensor(adj_matrix).float()
    # static_features_tensor = static_features.clone().detach()
    
    return adj_matrix_tensor, static_features

if __name__ == "__main__":
    main()
