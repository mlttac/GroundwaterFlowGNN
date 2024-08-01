# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:53:21 2024

@author: cnmlt
"""
import torch
from torch.utils.data import Dataset

class AutoregressiveTimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, max_future_window, missing_data_mask, num_piezo):
        self.data = data
        self.input_window = input_window
        self.max_future_window = max_future_window
        self.total_nodes = len(data.columns)
        self.missing_data_mask = missing_data_mask
        self.num_piezo = num_piezo  # Now num_piezo is passed as a parameter


    def __len__(self):
        return len(self.data) - self.input_window - self.max_future_window

    def __getitem__(self, idx):
        input_sequence = self.data.iloc[idx:(idx + self.input_window), :self.num_piezo]
        external_forces_sequence = self.data.iloc[idx:(idx + self.input_window + self.max_future_window), self.num_piezo:self.total_nodes]
        target_sequence = self.data.iloc[(idx + self.input_window):(idx + self.input_window + self.max_future_window), :self.num_piezo]

        # Generate mask sequence for the target sequence
        mask_sequence = self.missing_data_mask.iloc[(idx + self.input_window):(idx + self.input_window + self.max_future_window), :self.num_piezo]

        return (
            torch.tensor(input_sequence.values, dtype=torch.float32),
            torch.tensor(external_forces_sequence.values, dtype=torch.float32),
            torch.tensor(target_sequence.values, dtype=torch.float32),
            torch.tensor(mask_sequence.values, dtype=torch.float32)  # Add the mask tensor
        )