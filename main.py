import argparse
from data_preprocessing import process_data 
from data_preprocessing import gnn_data_prep
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import train_model import train_and_evaluate_model


def main():

    # Process data
    train_data, val_data, test_data, train_mask, val_mask, test_mask, df_piezo_columns, pump_columns, locations_no_missing, scaler = process_data.main()
    # Prepare GNN data
    # gnn_data_prep.main(df_piezo_columns, pump_columns, locations_no_missing)
    train_and_evaluate_model(train_data, val_data, test_data, df_piezo_columns, pump_columns, locations_no_missing)


if __name__ == "__main__":
    main()



