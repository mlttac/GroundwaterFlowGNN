
# Functions for data normalization, plotting distributions, and statistics calculation
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch 
from sklearn.metrics import mean_squared_error



# Function to calculate MSE for each piezometer
def calculate_mse_per_piezometer(predicted, target, num_piezometers):
    mse_values = []
    for i in range(num_piezometers):
        mse = mean_squared_error(target[:, i], predicted[:, i])
        mse_values.append(mse)
    return mse_values


def calculate_rmse_per_piezometer(predicted, target, num_piezometers):
    rmse_values = []
    for i in range(num_piezometers):
        mse = mean_squared_error(target[:, i], predicted[:, i])
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)
    return rmse_values



def calculate_rmse_per_piezometer_moria(df_moria, test_targets, df_piezo_columns):

    piezo_to_index = {name: index for index, name in enumerate(df_piezo_columns)}
    
    rmse_values = {}
    # Iterate over piezometer names in df_moria
    for piezo_name in df_moria.columns:
        if piezo_name in piezo_to_index:
            # Get the index of the piezometer in the test_targets array
            index = piezo_to_index[piezo_name]
            predicted = df_moria[piezo_name].values
            target = test_targets[:, index]
            # Ensure target and predicted are of the same length
            min_length = min(len(predicted), len(target))
            predicted = predicted[:min_length]
            target = target[:min_length]
            # Calculate MSE and then RMSE
            mse = mean_squared_error(target, predicted)
            rmse = np.sqrt(mse)
            rmse_values[piezo_name] = rmse
    
    return rmse_values



def calculate_bias_per_piezometer(predicted, target, num_piezometers):
    bias_values = []
    for i in range(num_piezometers):
        # Convert to NumPy array if they are PyTorch tensors
        if torch.is_tensor(predicted):
            predicted_np = predicted.cpu().detach().numpy()
        else:
            predicted_np = predicted

        if torch.is_tensor(target):
            target_np = target.cpu().detach().numpy()
        else:
            target_np = target

        # Calculate bias
        bias = np.sum(target_np[:, i] - predicted_np[:, i]) / np.sum(target_np[:, i])
        bias_values.append(bias)
    return bias_values




# Function to calculate mean and standard deviation
def print_mean_std(values, metric_name):
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric_name} - Mean: {mean_val:.4f}, Standard Deviation: {std_val:.4f}")
    return mean_val, std_val