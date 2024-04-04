import torch
from .datasets import SkinCancerDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import time
import os
import argparse
import itertools
from sklearn.model_selection import KFold
from .utils import evaluate_teacher, train_teacher_model


SKIN_CANCER_DATASET_DIR = '/home/dylanz/eecs592/data/skin_cancer'
CROSS_VAL_RESULTS_DIRECTORY = '/home/dylanz/eecs592/results/cross_val'
NUM_FOLDS = 5

# Train on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using the following device: " + str(device))

# Detects any issues with back propogation
torch.autograd.set_detect_anomaly(True)


# Generate command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    type=int,
                    help='Batch size for all models')
parser.add_argument('--num_epochs',
                    type=int,
                    help='Number of epochs for all models')
parser.add_argument('--teacher_lr',
                    type=float,
                    nargs='*',
                    default=[],
                    help='Learning rate for the teacher')
parser.add_argument('--teacher_l2_cost',
                    type=float,
                    nargs='*',
                    default=[],
                    help='L2 cost for the teacher')

# Parse command line arguments
args = parser.parse_args()
batch_size = args.batch_size
num_epochs = args.num_epochs
teacher_lr = list(args.teacher_lr)
teacher_l2_cost = list(args.teacher_l2_cost)


# Create dataframe to store the results and iterator of all hyperparameters to evaluate
columns = ['teacher_lr', 'teacher_l2_cost']
hp_iterator = {'teacher_lr': teacher_lr, 'teacher_l2_cost': teacher_l2_cost}

# Keep track of results
columns.append('score')

# Add all columns to dataframe
results_df = pd.DataFrame(columns = columns)



# Create hyperparameter iterator
keys, values = zip(*hp_iterator.items())
hp_iterator = [dict(zip(keys, v)) for v in itertools.product(*values)]


# Iterate over multiple different simulations
for hp_args in tqdm(hp_iterator, position=0, desc='Analysis'):

    # Obtain dataset
    csv_path = os.path.join(SKIN_CANCER_DATASET_DIR, 'training_0', 'training.csv')
    full_df = pd.read_csv(csv_path)

    # Get k-fold dataset
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    for fold, (train_ind, val_ind) in enumerate(kfold.split(full_df)):

        # Get k-fold training data
        train_k_df = full_df.iloc[train_ind]
        val_df = full_df.iloc[val_ind]

        # Get k-fold datasets
        train_k_dataset = SkinCancerDataset(train_k_df)
        val_dataset = SkinCancerDataset(val_df)

        # Get k-fold data loaders
        train_k_loader = DataLoader(train_k_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        # Train the teacher model
        teacher_model = train_teacher_model(train_k_loader,
                                            hp_args.get('teacher_lr'),
                                            hp_args.get('teacher_l2_cost'),
                                            num_epochs)

        # Evaluate the model
        score = evaluate_teacher(teacher_model, val_loader)
    
        # Save results to dataframe
        new_row = {}
        new_row['score'] = score
        new_row['teacher_lr'] = hp_args.get('teacher_lr')
        new_row['teacher_l2_cost'] = hp_args.get('teacher_l2_cost')

        results_df.loc[len(results_df)] = new_row
        print(results_df)


# Process results
file_name = 'teacher' + '_' + str(time.time())
results_df = results_df.groupby(['teacher_lr', 'teacher_l2_cost'], as_index=False).mean()
results_df.to_csv(os.path.join(CROSS_VAL_RESULTS_DIRECTORY, file_name), index=False)
print("Results: ")
print(results_df)