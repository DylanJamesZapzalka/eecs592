import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = '/home/dylanz/eecs592/data/raw_data'
TRAINING_SIZE = 0.8



if __name__ == "__main__":

    # Make sure directories exist
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError('Directory ' + RAW_DATA_DIR + ' does not exist.')

    # Load the raw  dataset
    csv_path = os.path.join(RAW_DATA_DIR, 'HAM10000_metadata')
    csv_df = pd.read_csv(csv_path)

    # Split train and test sets
    train_df, test_df = train_test_split(csv_df, train_size=TRAINING_SIZE, random_state=42)

    # Save CSVs
    train_df.to_csv(os.path.join(RAW_DATA_DIR, 'train_skin_cancer.csv'), index=False)
    test_df.to_csv(os.path.join(RAW_DATA_DIR, 'test_skin_cancer.csv'), index=False)