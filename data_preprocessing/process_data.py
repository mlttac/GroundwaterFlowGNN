# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:35:06 2024

@author: cnmlt
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import hashlib

from .preprocessing import (
    piezometer_measurements,
    resample_df,
    select_nodes,
    read_and_process_data, 
    aggregate_piezo_data )


from utils.visualization import plot_two_random_columns 


def setup_environment():
    """
    Setup the necessary environment, such as changing working directories and ensuring required folders exist.
    """
    os.chdir(Path(__file__).parent.parent.parent)
    print("Current working directory:", os.getcwd())  # Add this line

    # path_images = Path('imgs/')
    # path_images.mkdir(parents=True, exist_ok=True)
    preprocessed = Path("scripts/data")
    assert preprocessed.is_dir(), "Preprocessed data directory does not exist."

def load_and_filter_series(aquifer=None):
    """
    Load piezometer series metadata and filter based on criteria.
    """
    df = pd.read_csv('data/piezometers/oseries_metadata_and_selection.csv')
    if aquifer:
        return df[df['aquifer'] == aquifer]['name']
    else:
        return df['name']

def process_series(series_names, start_year=2004, resampling_freq='W'):
    """
    Process each series in the list of series names.
    """
    dfs = []
    for series_name in series_names:
        try:
            df_series = piezometer_measurements(series_name)
            if df_series is not None:
                df_series = df_series.groupby('date').agg({'HEAD': 'mean'})
                df_series = df_series.resample(resampling_freq).mean()
                dfs.append(df_series.rename(columns={'HEAD': series_name}))
        except FileNotFoundError as e:
            print(f"Error processing series {series_name}")

    complete_daily = pd.concat(dfs, axis=1)
    complete_daily = complete_daily[complete_daily.index.year >= start_year]
    return complete_daily


def save_data_with_config_hash(data, config, data_filepath, config_hash_filepath):
    """
    Save data with configuration hash to check for changes.

    Parameters:
    - data: The data to be saved.
    - config: The configuration settings used for processing the data.
    - data_filepath: The file path for saving the processed data.
    - config_hash_filepath: The file path for saving the configuration hash.
    """
    # Serialize data and save
    with open(data_filepath, 'wb') as f:
        pickle.dump(data, f)
    
    # Serialize config hash and save
    config_hash = hashlib.sha256(pickle.dumps(config)).hexdigest()
    with open(config_hash_filepath, 'wb') as f:
        pickle.dump(config_hash, f)


def load_data_if_config_unchanged(config, base_data_path):
    """Load data if configuration hasn't changed."""
    data_type_prefix = "_synthetic" if config['synthetic_data'] else ""
    
    processed_data_filepath = base_data_path / f'processed_data{data_type_prefix}.pkl'
    config_hash_filepath = base_data_path / 'config_hash{data_type_prefix}.pkl'
    
    # Check if config hash file exists and compare with current config
    try:
        with open(config_hash_filepath, 'rb') as f:
            saved_config_hash = pickle.load(f)
        current_config_hash = hashlib.sha256(pickle.dumps(config)).hexdigest()
        
        if saved_config_hash == current_config_hash:
            with open(processed_data_filepath, 'rb') as f:
                print("Loading data from saved file.")
                data = pickle.load(f)
                return data  # Returns the tuple of data (complete_daily, df_piezo, missing_data_mask)
    except FileNotFoundError:
        print("Config hash or processed data file not found. Processing data.")
    
    return None


def save_data(data, data_filepath):
    """
    Save data to a specified file path.

    Parameters:
    - data: The data to be saved.
    - data_filepath: The file path for saving the processed data.
    """
    # Serialize data and save
    with open(data_filepath, 'wb') as f:
        pickle.dump(data, f)

    print("Data saved successfully.")

def load_data(base_data_path, config):
    """
    Load data from a specified file path.
    
    Parameters:
    - base_data_path: The base directory where data files are stored.
    - config: The configuration settings used for determining the data file path.
    
    Returns:
    - Loaded data if the file exists, otherwise None.
    """
    data_type_prefix = "_synthetic" if config['synthetic_data'] else ""
    processed_data_filepath = base_data_path / f'processed_data{data_type_prefix}.pkl'
    
    try:
        with open(processed_data_filepath, 'rb') as f:
            print("Loading data from saved file.")
            data = pickle.load(f)
            return data
    except FileNotFoundError:
        print("Processed data file not found. Processing data.")
    
    return None



def save_column_names(columns, base_data_path, data_type_suffix=''):
    """
    Save column names to a text file, appending a suffix to differentiate between real and synthetic data.

    Parameters:
    - columns: Iterable containing the column names.
    - base_data_path: Path object pointing to the base data directory.
    - data_type_suffix: String indicating the data type ("_synthetic" for synthetic data, "_real" for real data, otherwise an empty string).
    """
    column_names_filepath = base_data_path / f'column_names{data_type_suffix}.txt'
    
    with open(column_names_filepath, 'w') as file:
        for column in columns:
            file.write(column + '\n')


def load_column_names(base_data_path, data_type_suffix=''):
    """
    Load column names from a text file if it exists, appending a suffix to differentiate between real and synthetic data.

    Parameters:
    - base_data_path: Path object pointing to the base data directory.
    - data_type_suffix: String indicating the data type ("_synthetic" for synthetic data, "_real" for real data, otherwise an empty string).

    Returns:
    - A list of column names if the file exists and is loaded successfully, otherwise None.
    """
    column_names_filepath = base_data_path / f'column_names{data_type_suffix}.txt'
    
    # Check if the column names file exists
    if column_names_filepath.exists():
        with open(column_names_filepath, 'r') as file:
            # Read column names into a list, stripping newline characters
            columns = [line.strip() for line in file.readlines()]
        return columns
    else:
        # Return None if the file does not exist
        return None


def fill_and_select_data(df, n_nodes_selection=200):
    """
    Fill missing data and select nodes based on selection criteria.
    """
    
    df =  select_nodes(df, n_nodes_selection)
    
    missing_data_mask = ~df.isna()

    df = df.interpolate(method='linear', order=3).bfill(limit=None)


    return df, missing_data_mask



def process_synthetic_data(synthetic_data_path, real_data_path, series_names, config):
    """
    Incorporate reading, processing, aggregating, deduplicating, and resampling of synthetic data.
    """
    df_piezo_moria = aggregate_piezo_data(synthetic_data_path)
    print(f"Unique columns after deduplication: {len(df_piezo_moria.columns)}")

    filtered_columns = [col for col in series_names if col in df_piezo_moria.columns]
    print(f"Filtered columns count: {len(filtered_columns)}")

    additional_needed = config['n_nodes_selection'] - len(filtered_columns)
    if additional_needed > 0:
        # check columns that are present for both in the metadata and df_piezo_moria, excluding already filtered columns

        piezo_metadata = pd.read_csv(real_data_path/ "piezometers/oseries_metadata_and_selection.csv")  
        common_columns = piezo_metadata[piezo_metadata['name'].isin(df_piezo_moria.columns)]
        potential_additional_columns = common_columns[common_columns['name'].isin(filtered_columns) == False]['name'].tolist()
        # Select randomly without repetition
        selected_randomly = np.random.choice(potential_additional_columns, size=min(additional_needed, len(potential_additional_columns)), replace=False)
        selected_columns = filtered_columns + list(selected_randomly)
    else:
        selected_columns = filtered_columns[:config['n_nodes_selection']]

    df_piezo_moria_selected = df_piezo_moria[selected_columns]
    df_piezo_moria_unique_cols = df_piezo_moria_selected.loc[:, ~df_piezo_moria_selected.columns.duplicated(keep='first')]

    common_start_date = pd.Timestamp(config.get('common_start_date', '2008-01-01'))
    df_piezo_moria_unique_cols.index = pd.to_datetime(df_piezo_moria_unique_cols.index)
    resampled_df = df_piezo_moria_unique_cols.resample(config['resampling_freq'], origin=common_start_date).mean()

    missing_data_mask = ~resampled_df.isna()
    return resampled_df, missing_data_mask



def load_external_data(data_path, common_start_date, common_end_date, resampling_freq):
    """
    Load and preprocess external data sources including pumping wells, precipitation, evaporation, and river data.
    
    Parameters:
    - data_path: Path to the directory containing external data files.
    - common_start_date: The start date for filtering and resampling the data.
    - common_end_date: The end date for filtering and resampling the data.
    - resampling_freq: Frequency for resampling the time series data.
    
    Returns:
    - Tuple of DataFrames: (df_pumping_wells, df_precipitation, df_evaporation, df_river)
    """
    # Load pumping wells data
    df_pumping_wells = pd.read_csv(data_path / "wells/wells_daily_preprocessed.csv", index_col='Datum', parse_dates=True)
    df_pumping_wells.index = pd.to_datetime(df_pumping_wells.index)
    df_pumping_wells = resample_df(df_pumping_wells, common_start_date, common_end_date, resampling_freq ).fillna(0)
    
    # Load precipitation data
    df_precipitation = pd.read_csv(data_path / "meteo_metadata_and_timeseries/precipitation.csv")
    df_precipitation = df_precipitation.rename(columns={'Unnamed: 0': 'Datum'})
    df_precipitation['Datum'] = pd.to_datetime(df_precipitation['Datum'])
    df_precipitation = df_precipitation.set_index('Datum')
    df_precipitation = resample_df(df_precipitation, common_start_date, common_end_date, resampling_freq )
    
    # Load evaporation data
    df_evaporation = pd.read_csv(data_path / "meteo_metadata_and_timeseries/evaporation.csv")
    df_evaporation = df_evaporation.rename(columns={'Unnamed: 0': 'Datum'})
    df_evaporation['Datum'] = pd.to_datetime(df_evaporation['Datum'])
    df_evaporation = df_evaporation.set_index('Datum')
    df_evaporation = resample_df(df_evaporation, common_start_date, common_end_date, resampling_freq )
    
    # Load river data
    df_river = pd.read_csv(data_path / "river/river_daily.csv")
    df_river = df_river.rename(columns={'Unnamed: 0': 'Datum'})
    df_river['Datum'] = pd.to_datetime(df_river['Datum'])
    # Set the index to 'Datum'
    df_river = df_river.set_index('Datum')
    df_river = resample_df(df_river, common_start_date, common_end_date, resampling_freq )

    return df_pumping_wells, df_precipitation, df_evaporation, df_river



def split_and_normalize_data(df_piezo, missing_data_mask, external_data, config):
    """
    Splits the dataset into training, validation, and test sets, normalizes them, and also splits the missing data mask.

    Parameters:
    - df_piezo: DataFrame containing piezometer data.
    - missing_data_mask: DataFrame indicating the presence of missing data.
    - external_data: A tuple containing DataFrames of external data (pumping wells, precipitation, evaporation, river).
    - config: A dictionary with configuration settings such as split ratios.

    Returns:
    - A tuple of normalized DataFrames: (train_data, val_data, test_data, train_mask, val_mask, test_mask).
    """
    # Combine piezometer data with external data for processing
    combined_data = pd.concat([df_piezo] + list(external_data), axis=1)

    test_val_size = 0.2
    # Split data into train, validation, and test sets initially
    train_data, temp_data = train_test_split(combined_data, test_size=test_val_size, random_state=42, shuffle=False)
    # val_data, test_data = train_test_split(temp_data, test_size=test_val_split, random_state=42, shuffle=False)
    val_data = temp_data
    test_data = temp_data
    
    train_mask, temp_mask = train_test_split(missing_data_mask, test_size=test_val_size, random_state=42, shuffle=False)
    # val_mask, test_mask = train_test_split(temp_mask, test_size=test_val_split, random_state=42, shuffle=False)
    val_mask = temp_mask
    test_mask = temp_mask
    # Normalize the datasets
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = pd.DataFrame(scaler.fit_transform(train_data), index=train_data.index, columns=train_data.columns)
    val_data = pd.DataFrame(scaler.transform(val_data), index=val_data.index, columns=val_data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), index=test_data.index, columns=test_data.columns)

    # Return the normalized datasets along with their corresponding masks
    return train_data, val_data, test_data, train_mask, val_mask, test_mask, scaler


def define_configuration(synthetic_data):
    return {
        'start_year': 2004,
        'resampling_freq': 'W',
        'n_nodes_selection': 200,
        'aquifer': None,
        'synthetic_data': synthetic_data,  
        'plotting': True, 

    }



def main(synthetic_data):
    config = define_configuration(synthetic_data)
    
    base_data_path = Path('C:/codes/wells_time/scripts/data/preprocessed/').absolute()
    data_path = Path('C:/codes/wells_time/scripts/data/input/')
    
    # File paths for saving/loading processed data and configuration hash
    # Adjust file paths based on whether data is synthetic
    data_type_prefix = "_synthetic" if config['synthetic_data'] else ""
    processed_data_filepath = base_data_path / f'processed_data{data_type_prefix}.pkl'
    config_hash_filepath = base_data_path / f'config_hash{data_type_prefix}.pkl'

    data = load_data(base_data_path, config)

    common_start_date = pd.Timestamp('2008-01-01') if config['synthetic_data'] else pd.Timestamp('2004-01-01')
    
    if data is not None:
        # Unpack loaded data
        df_piezo, missing_data_mask = data
        print("Data loaded from saved file.")
    else:
        print("Processing new data.")
        setup_environment()

        if config['synthetic_data']:
            
            series_names_real = load_column_names(base_data_path, '_real' )
            synthetic_data_path = Path('scripts/data/simulated_data/results/timeseries/')
            df_piezo, missing_data_mask = process_synthetic_data(synthetic_data_path, data_path, series_names_real, config)
            save_column_names(df_piezo.columns, base_data_path, '_synthetic')

        else:
            series_names = load_and_filter_series(config['aquifer'])
            complete_daily = process_series(series_names, config['start_year'], config['resampling_freq'])
            df_piezo, missing_data_mask = fill_and_select_data(complete_daily, config['n_nodes_selection'])
            save_column_names(df_piezo.columns, base_data_path, '_real')

        common_end_date = df_piezo.index[-1]

        df_pumping_wells, df_precipitation, df_evaporation, df_river = load_external_data(data_path, common_start_date, common_end_date, config['resampling_freq'])

        # Save processed data and config hash
        data_to_save = (df_piezo, missing_data_mask)
        # save_data_with_config_hash(data_to_save, config, processed_data_filepath, config_hash_filepath)
        save_data(data_to_save, processed_data_filepath)



    common_end_date = df_piezo.index[-1]

    df_pumping_wells, df_precipitation, df_evaporation, df_river = load_external_data(data_path, common_start_date, common_end_date, config['resampling_freq'])

    # Example definitions based on processed data
    df_piezo_columns = df_piezo.columns.tolist()  # Piezometer column names
    pump_columns = df_pumping_wells.columns.tolist()  # Pump column names

    exclude_locations = ['Driel beneden', 'Driel boven', 'Arnhem']
    df_river = df_river.drop(columns=exclude_locations, errors='ignore')

    # Count the missing data for each location
    missing_counts = df_river.isna().sum()
    
    locations_no_missing = missing_counts[missing_counts == 0].index
    df_river = df_river[locations_no_missing]

    train_data, val_data, test_data, train_mask, val_mask, test_mask, scaler = split_and_normalize_data(df_piezo, missing_data_mask, (df_pumping_wells, df_precipitation, df_evaporation, df_river), config)

    # Return the additional variables alongside the datasets
    return train_data, val_data, test_data, train_mask, val_mask, test_mask, df_piezo_columns, pump_columns, locations_no_missing, scaler

if __name__ == "__main__":
    main()


