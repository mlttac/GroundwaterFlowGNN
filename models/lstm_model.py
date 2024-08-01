import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, external_forces_size, dense_output_size):
        super(LSTMModel, self).__init__()
        self.dense = nn.Linear(external_forces_size, dense_output_size)
        self.lstm1 = nn.LSTM(input_size + dense_output_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, external_forces):

        input_seq = input_seq.squeeze(1)  # This changes the shape to [32, 219, 6]

        batch_size, seq_len, _ = external_forces.shape
        aligned_forces = self.dense(external_forces.reshape(-1, external_forces.size(-1)))
        aligned_forces = aligned_forces.reshape(batch_size, seq_len, -1)

        # Swap the dimensions of input_seq to align the temporal dimension
        input_seq_swapped = input_seq.transpose(1, 2)  # Swaps the second and third dimensions

        # Concatenate input_seq_swapped and aligned_forces along the feature dimension
        combined_input = torch.cat((input_seq_swapped, aligned_forces), dim=2)

        # LSTM layers
        x, _ = self.lstm1(combined_input)
        x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)

        # Output layer for only the last time step
        x = self.output_layer(x[:, -1, :])
 

        # Add an extra time dimension to match the target shape
        return x.unsqueeze(1)
    
def create_lstm_model():
    input_size = 219  
    hidden_size = 150
    output_size = 200  
    external_forces_size = 19
    dense_output_size = 100

    # Correct input size calculation
    return LSTMModel(input_size, hidden_size, output_size, external_forces_size, dense_output_size)

