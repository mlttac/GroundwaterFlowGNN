# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:52:28 2024

@author: cnmlt
"""

import torch
import numpy as np
import os
import json
import pandas as pd


def prepare_combined_input(input_seq, external_forces, modeltype = 'MTGNN'):
    # Replicate the first time step of input_seq
    first_step_replicated = input_seq[:, 0, :].unsqueeze(1)
    input_seq_padded = torch.cat([first_step_replicated, input_seq], dim=1)

    # Concatenate padded input sequence with external forces along the feature dimension
    combined_input = torch.cat([input_seq_padded, external_forces], dim=2)

    if modeltype == 'MTGNN' or 'MTGNN_LSTM': 
        # Reshape combined input to match the expected input format of MTGNN
        # torch.Size([B, T, N]) --- > torch.Size([B, 1, N, T])
        combined_input = combined_input.permute(0, 2, 1).unsqueeze(1)
    else:
        combined_input = combined_input.permute(0, 2, 1).unsqueeze(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return combined_input.to(device)

def make_predictions(model, sample, device, F_w, W, A_tilde, static_features, num_piezo, build_adj=False, modeltype='MTGNN', perturb=False, noise_level=0.01):
    input_sequence, external_forces, target_sequence, _ = sample

    # Move the data to the device (CPU or CUDA)
    input_sequence = input_sequence.to(device)
    external_forces = external_forces.to(device)

    # Apply high-frequency noise perturbation if required
    if perturb:
        noise = torch.randn_like(external_forces) * noise_level
        external_forces += noise

    model.eval()
    current_input = input_sequence.unsqueeze(0)  # Add batch dimension
    current_external_forces = external_forces.unsqueeze(0)  # Add batch dimension

    predictions = []
    with torch.no_grad():  # Disable gradient computations
        for t in range(F_w):  # Iterate for F future steps
            current_forces = current_external_forces[:, t : (W + t + 1), :]
            # combined_input = prepare_combined_input(current_input, current_forces, modeltype)
            combined_input = prepare_combined_input(current_input, current_forces)

            if modeltype == 'MTGNN' or 'MTGNN_LSTM':
                output = model(combined_input, A_tilde.to(device), FE=static_features.to(device)) if not build_adj else model(combined_input, FE=static_features.to(device))
                output = output[:, :, :num_piezo, 0]
            else: 
                output = model(combined_input.to(device), current_forces.to(device)) 

            predictions.append(output)
            next_input = output 
            current_input = torch.cat((current_input[:, 1:, :], next_input), dim=1)

    predicted_sequence = torch.cat(predictions, dim=1).squeeze(0)  # Remove batch dimension

    # Move the predicted sequence back to CPU for further processing
    return input_sequence.cpu(), predicted_sequence.cpu(), target_sequence.cpu()


def inverse_transform_with_shape_adjustment(data, scaler, original_feature_count):
    """
    Adjusts the shape of the data for the scaler's inverse_transform method and extracts the original features.

    Parameters:
    - data: The data to inverse transform, as a numpy array.
    - scaler: The fitted scaler object used for inverse transformation.
    - original_feature_count: The number of original features in the data before scaling.

    Returns:
    - The data after inverse transformation, with only the original features.
    """
    # Check if data needs dummy features appended
    if data.shape[1] < scaler.n_features_in_:
        # Calculate the number of dummy features required
        dummy_feature_count = scaler.n_features_in_ - original_feature_count
        # Create a dummy array with the required number of dummy features
        dummy_features = np.zeros((data.shape[0], dummy_feature_count))
        # Append dummy features to the data
        data_with_dummy = np.hstack((data, dummy_features))
    else:
        data_with_dummy = data

    # Apply the inverse transformation
    data_inversed = scaler.inverse_transform(data_with_dummy)

    # Extract the original features
    data_inversed_corrected = data_inversed[:, :original_feature_count]

    return data_inversed_corrected

def generate_model_filename(modelname, future_window, **config):
    base_filename = f"model_{modelname}_F{future_window}"
    params_filename = "_".join([f"{value}" for key, value in config.items() if key not in ['modelname']])
    filename = f"saved_models/{base_filename}_{params_filename}.pt"
    # Replace square brackets and spaces in the filename
    filename = filename.replace('[', '').replace(']', '').replace(' ', '_')

    return filename



        

def save_rmse_values(test_rmse, model_type, future_window,**config):
    model_base_name = generate_model_filename(model_type, future_window,**config)
    
    model_base_name = os.path.basename(model_base_name).replace('.pt', '')

    os.makedirs("training_results", exist_ok=True)
    rmse_filename = f"{model_base_name}_rmse_test.json"
    rmse_filepath = os.path.join("training_results", rmse_filename)

    # Convert test_rmse to a list if it's not already one (e.g., if it's a numpy array)
    test_rmse_list = test_rmse if isinstance(test_rmse, list) else test_rmse.tolist()

    # Directly save or overwrite the RMSE values
    with open(rmse_filepath, 'w') as f:
        json.dump(test_rmse_list, f, indent=4)



results = []  # Assuming results is a list where you append dictionaries of configuration and performance metrics.

def record_result(config, test_rmse_mean, test_rmse_std):
    """
    Record the configuration and its performance metrics.
    """
    result = config.copy()
    result['test_rmse_mean'] = test_rmse_mean
    result['test_rmse_std'] = test_rmse_std
    results.append(result)

def make_hashable(df):
    """
    Convert any unhashable columns in the DataFrame to a hashable type.
    This specifically targets lists, converting them to tuples.
    """
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            df[column] = df[column].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

def analyze_results():
    """
    Analyze the recorded results to find the best configuration and the best values for each parameter.
    """
    df = pd.DataFrame(results)
    df = make_hashable(df)  # Ensure all data is hashable before analysis
    
    # Find the best overall configuration
    best_overall = df.loc[df['test_rmse_mean'].idxmin()]
    print("Best Overall Configuration:")
    print(best_overall, '\n')
    
    # Find the best values for each parameter
    parameters = [col for col in df.columns if col not in ['test_rmse_mean', 'test_rmse_std']]
    for param in parameters:
        grouped = df.groupby(param)['test_rmse_mean'].agg(['idxmin'])
        best_indices = grouped['idxmin']
        best_for_param = df.loc[best_indices][[param, 'test_rmse_mean']].drop_duplicates()
        print(f"Best {param}:")
        print(best_for_param.to_string(index=False), '\n')





import pandas as pd 
from pathlib import Path

from data_preprocessing.preprocessing import aggregate_piezo_data 
from data_preprocessing.process_data import define_configuration 


def get_synthetic(series_names):
    config = define_configuration(True)
    
    synthetic_data_path = Path('C:/codes/wells_time/data/simulated_data/results/timeseries/').absolute()
    moria_results_overview_path = synthetic_data_path / 'moria_results_overview.csv'
    
    # Check if the moria_results_overview.csv file already exists
    if moria_results_overview_path.exists():
        print(f"File {moria_results_overview_path} exists. Loading...")
        resampled_df = pd.read_csv(moria_results_overview_path, index_col=0, parse_dates=True)
        missing_data_mask = ~resampled_df.isna()
        return resampled_df, missing_data_mask
    
                
    df_piezo_moria = aggregate_piezo_data(synthetic_data_path)
    print(f"Unique columns after deduplication: {len(df_piezo_moria.columns)}")
    
    filtered_columns = [col for col in series_names if col in df_piezo_moria.columns]
    print(f"Filtered columns count: {len(filtered_columns)}")
    
    df_piezo_moria_selected = df_piezo_moria[filtered_columns]
    df_piezo_moria_unique_cols = df_piezo_moria_selected.loc[:, ~df_piezo_moria_selected.columns.duplicated(keep='first')]
    
    df_piezo_moria_unique_cols.index = pd.to_datetime(df_piezo_moria_unique_cols.index)
    resampled_df = df_piezo_moria_unique_cols.resample(config['resampling_freq']).mean()
    
    resampled_df = resampled_df[resampled_df.index >= pd.Timestamp('2018-04-01')]

    # Only select the first 100 rows of resampled_df
    resampled_df_first_100 = resampled_df.head(100)
    
    # Save to CSV
    resampled_df_first_100.to_csv(moria_results_overview_path)
    print(f"File saved to {moria_results_overview_path}")
    
    missing_data_mask = ~resampled_df_first_100.isna()
    
    return resampled_df_first_100, missing_data_mask

