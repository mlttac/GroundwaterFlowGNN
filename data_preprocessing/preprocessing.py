
# Import statements and directory setup
import os
import pandas as pd 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io
import glob

from sklearn.preprocessing import MinMaxScaler



def piezometer_measurements(series_name):
    # Split series_name with handling for potential extra dashes
    parts = series_name.split('-')
    if len(parts) >= 2:
        locatie = parts[0]
        filternummer = '-'.join(parts[1:])
    else:
        warnings.warn(f"Unexpected format for series_name: {series_name}")
        return None

    # create a pattern for the filenames
    pattern = os.path.join('data', 'piezometers', 'csv', 'csv', f'{locatie}{filternummer}*')

    if not any(glob.glob(pattern)):
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in the directory.")

    for filename in glob.glob(pattern):
        # find the start and end of the first and second data block
        with open(filename, 'r') as f:
            lines = f.readlines()
            start_index = []
            end_index = []
            for i, line in enumerate(lines):
                if line.startswith('LOCATIE,FILTERNUMMER'):
                    if len(start_index) < 2:
                        start_index.append(i)
                    if len(start_index) > 1 and len(end_index) < 1:
                        end_index.append(i)
                elif len(start_index) > 0 and i > 30:
                    end_index.append(i)
                    break

        if len(start_index) > 0 and len(end_index) > 0:
            # check the first block of data
            data_block = io.StringIO(''.join(lines[start_index[0]:end_index[0]]))
            df0 = pd.read_csv(data_block, header=0, dtype={'FILTERNUMMER': str})

            # check that LOCATIE and FILTERNUMMER are correct
            for index, row in df0.iterrows():
                if not (row['LOCATIE'] == locatie and row['FILTERNUMMER'] == filternummer):
                    print(row['LOCATIE'], locatie , row['FILTERNUMMER'] , filternummer)
                    print(f"Data inconsistency in file {filename} at line {index + start_index[0] + 1}") # +1 due to the index starting at 0

            # if there is a second block, store it into a DataFrame
            if len(start_index) > 1:
                data_block = io.StringIO(''.join(lines[start_index[1]:]))
                df = pd.read_csv(data_block, header=0, dtype={'FILTERNUMMER': str})

    df.rename(columns={"STAND (cm NAP)": "HEAD"}, inplace=True)
    df['date'] = pd.to_datetime(df['PEIL DATUM TIJD'])


    # Convert 'BIJZONDERHEID' values to string and strip spaces
    df['BIJZONDERHEID'] = df['BIJZONDERHEID'].astype(str).str.strip()

    # Filter rows where 'BIJZONDERHEID' is 'reliable'
    df = df[df['BIJZONDERHEID'] == 'reliable']

    # Print unique values in 'BIJZONDERHEID' that are neither 'reliable' nor 'unreliable'
    other_values = df.loc[~df['BIJZONDERHEID'].isin(['reliable', 'unreliable']), 'BIJZONDERHEID'].unique()
    if len(other_values) > 0:
        print(f"Other unique values found in 'BIJZONDERHEID': {other_values}")

    # df.set_index("date", inplace=True)

    return df






def select_nodes(complete_daily, n_nodes_selection):
    """
    # The `select_nodes` function selects nodes (piezometers) with the least amount of missing data (`NaN` values) from a given DataFrame. 
    
    Parameters:
    - complete_daily: DataFrame containing the data
    - n_nodes_selection: Number of nodes to select
    
    Returns:
    - filtered_df: DataFrame containing only the selected nodes
    """
    
    # Calculate the percentage of NaN values per column for the filtered data
    nan_percentages = complete_daily.isnull().mean()

    # Sort the nodes based on their NaN percentages
    sorted_nodes = nan_percentages.sort_values().index[:n_nodes_selection]

    # Filter the original dataframe to only include the selected nodes
    filtered_df = complete_daily[sorted_nodes]
    
    return filtered_df




def resample_df(df_series, common_start_date, common_end_date, resampling_freq ):
    df_series_2D = df_series.resample(resampling_freq, origin=common_start_date).mean()
    new_index = pd.date_range(start=common_start_date, end=common_end_date, freq=resampling_freq)
    df_series_2D = df_series_2D.reindex(new_index)
    return df_series_2D


def read_and_process_data(file_path):
    """
    Read and process individual data files, converting measurement values and handling missing data.
    """
    data = pd.read_csv(file_path, skiprows=6, delim_whitespace=True, header=None, usecols=[0, 1, 2])
    data.columns = ['Date', 'Measurement', 'Computed_MORIA']
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d%H%M%S')
    data.set_index('Date', inplace=True)
    data['Measurement'] = data['Measurement'].replace(-9999.0000000, np.nan) * 100
    data['Computed_MORIA'] = data['Computed_MORIA'] * 100
    return data


def aggregate_piezo_data(data_path):
    """
    Aggregate data from all piezometer files within a specified directory into a single DataFrame.
    """
    df_piezo_moria = pd.DataFrame()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = Path(root) / file
                data = read_and_process_data(file_path)
                piezometer_name = file[:-4]
                if 'Computed_MORIA' in data:
                    df_piezo_moria = pd.concat([df_piezo_moria, data['Computed_MORIA'].rename(piezometer_name)], axis=1)
    df_piezo_moria = df_piezo_moria.align(df_piezo_moria, join='outer')[0]
    return df_piezo_moria


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def print_combined_statistics(df, df_name):
    combined_data = df.values.flatten()
    print(f"{df_name} Combined Statistics:")
    print(f"Mean: {np.mean(combined_data):.2f}")
    print(f"Median: {np.median(combined_data):.2f}")
    print(f"Standard Deviation: {np.std(combined_data):.2f}")
    print(f"Min: {np.min(combined_data):.2f}")
    print(f"Max: {np.max(combined_data):.2f}\n")




