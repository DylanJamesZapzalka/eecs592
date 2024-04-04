import torch
from .datasets import SkinCancerDataset, TeacherDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import time
import os
import argparse
import itertools
from .utils import train, evaluate, train_teacher_model
from .utils import get_teacher_logits, evaluate_worst_group_accuracy
from sklearn.model_selection import KFold
from transformers import BertTokenizer


# Train on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using the following device: " + str(device))

# Detects any issues with back propogation
torch.autograd.set_detect_anomaly(True)

SKIN_CANCER_DATASET_DIR = '/home/dylanz/eecs592/data/skin_cancer'
CROSS_VAL_RESULTS_DIRECTORY = '/home/dylanz/eecs592/results/cross_val'
NUM_FOLDS = 5

# Number of workers
NUM_WORKERS = 10


# Generate command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',
                    type=str,
                    help='Type of model to train -- can be \"l2\", \"tipmi\", \"tipmi-cf\", \"ban\", \"gdro\", or \"mmd\" ')
parser.add_argument('--num_datasets',
                    type=int,
                    help='Number of datasets to perform cross validation over.')
parser.add_argument('--batch_size',
                    type=int,
                    help='Batch size for all models')
parser.add_argument('--num_epochs',
                    type=int,
                    help='Number of epochs for all models')
parser.add_argument('--lr',
                    type=float,
                    nargs='*',
                    default=[],
                    help='Learning rate for l2, mmd, or gdro model')
parser.add_argument('--l2_cost',
                    type=float,
                    nargs='*',
                    default=[],
                    help='L2 cost for the l2, mmd, or gdro model')
parser.add_argument('--student_lr',
                    type=float,
                    nargs='*',
                    default=[],
                    help='Learning rate for the student')
parser.add_argument('--student_l2_cost',
                    type=float,
                    nargs='*',
                    default=[],
                    help='L2 cost for the student')
parser.add_argument('--teacher_lr',
                    type=float,
                    nargs='?',
                    default=0,
                    help='Learning rate for the teacher')
parser.add_argument('--teacher_l2_cost',
                    type=float,
                    nargs='?',
                    default=0,
                    help='L2 cost for the teacher')
parser.add_argument('--sigma',
                    type=float,
                    nargs='*',
                    default=[],
                    help='Kernel bandwidth for MMD')
parser.add_argument('--alpha',
                    type=float,
                    nargs='*',
                    default=[],
                    help='Cost of MMD regularization')

# Parse command line arguments
args = parser.parse_args()
model_name = args.model_name
num_datasets = args.num_datasets
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = list(args.lr)
l2_cost = list(args.l2_cost)
student_lr = list(args.student_lr)
student_l2_cost = list(args.student_l2_cost)
teacher_lr = args.teacher_lr
teacher_l2_cost = args.teacher_l2_cost
sigma = list(args.sigma)
alpha = list(args.alpha)


# Create dataframe to store the results and iterator of all hyperparameters to evaluate
if model_name == 'l2' or model_name == 'gdro':
    columns = ['lr', 'l2_cost']
    hp_iterator = {'lr': lr, 'l2_cost': l2_cost}

elif model_name == 'mmd':
    columns = ['lr', 'sigma', 'alpha']
    hp_iterator = {'lr': lr, 'sigma': sigma, 'alpha': alpha}

elif model_name == 'tipmi' or model_name == 'tipmi-cf' or model_name == 'ban':
    columns = ['student_lr', 'student_l2_cost']
    hp_iterator = {'student_lr': student_lr, 'student_l2_cost': student_l2_cost}

# Keep track of results
columns.append('training_set')
columns.append('score')
columns.append('worst_group_score')

# Add all columns to dataframe
results_df = pd.DataFrame(columns = columns)



# Create hyperparameter iterator
keys, values = zip(*hp_iterator.items())
hp_iterator = [dict(zip(keys, v)) for v in itertools.product(*values)]


# Iterate over multiple different simulations
for random_seed in tqdm(range(num_datasets), position=0, desc='Random Seeds'):

    # Obtain dataset
    csv_path = os.path.join(SKIN_CANCER_DATASET_DIR, f'training_{random_seed}', 'training.csv')
    full_df = pd.read_csv(csv_path)

    train_df = full_df[0: int(len(full_df) * 0.8)]
    test_df = full_df[int(len(full_df) * 0.8) :]

    # Get datasets
    include_extra = model_name == 'tipmi' or model_name == 'tipmi-cf' or model_name == 'ban'
    include_group = model_name == 'gdro'
    include_aux = model_name == 'mmd'
    train_dataset = SkinCancerDataset(train_df, include_group=include_group, include_aux=include_aux)
    val_dataset = SkinCancerDataset(train_df, include_extra=include_extra)
    test_dataset = SkinCancerDataset(test_df)
    test_group_dataset = SkinCancerDataset(test_df, include_group=True)

    # Create data loader for l2 or the teacher
    if model_name == 'l2' or model_name == 'teacher' or model_name == 'gdro' or model_name == 'mmd':

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

    # Create data loader for tipmi or ban
    elif model_name == 'tipmi':

        # Train the teacher model with a temporary data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        teacher_model = train_teacher_model(train_loader, teacher_lr, teacher_l2_cost, num_epochs)
        # Get logits
        extra_info, teacher_logits = get_teacher_logits(teacher_model, val_loader)
        # Get data loader
        train_dataset = TeacherDataset(extra_info, teacher_logits)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    

    # Create data loader for tipmi-cf
    elif model_name == 'tipmi-cf':

        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        extra_info_list = []
        teacher_logits_list = []
        for fold, (teacher_train_ind, teacher_val_ind) in enumerate(kfold.split(train_df)):

            # Get k-fold training data
            train_k_df = train_df.iloc[teacher_train_ind]
            val_df = train_df.iloc[teacher_val_ind]

            # Get k-fold datasets
            train_k_dataset = SkinCancerDataset(train_k_df)
            val_dataset = SkinCancerDataset(val_df, include_extra=True)

            # Get k-fold data loaders
            train_k_loader = DataLoader(train_k_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

            # Train the teacher model
            teacher_model = train_teacher_model(train_k_loader, teacher_lr, teacher_l2_cost, num_epochs)
            # Get the teacher training data
            extra_info, teacher_logits = get_teacher_logits(teacher_model, val_loader)
            extra_info_list = extra_info_list + extra_info
            teacher_logits_list.append(teacher_logits)

        # Create the teacher data loader
        teacher_logits_list = torch.concat(teacher_logits_list, dim=0)
        train_dataset = TeacherDataset(extra_info_list, teacher_logits_list)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

    for hp_args in tqdm(hp_iterator, position=0, desc='Analysis'):


        # Train the model
        if model_name == 'l2':
            model = train(model_name,
                            num_epochs,
                            train_loader,
                            lr=hp_args.get('lr'),
                            l2_cost=hp_args.get('l2_cost')
                            )
        elif model_name == 'gdro':
            dataset_info = train_dataset.get_group_info()
            model = train(model_name,
                            num_epochs,
                            train_loader,
                            lr=hp_args.get('lr'),
                            l2_cost=hp_args.get('l2_cost'),
                            dataset_info=dataset_info
                            )
        elif model_name == 'mmd':
            mmd_info = {'sigma': hp_args.get('sigma'), 'alpha': hp_args.get('alpha')}
            model = train(model_name,
                            num_epochs,
                            train_loader,
                            lr=hp_args.get('lr'),
                            l2_cost=0,
                            mmd_info=mmd_info
                            )  
        elif model_name == 'tipmi' or model_name == 'tipmi-cf' :
            model = train(model_name,
                            num_epochs,
                            train_loader,
                            lr=hp_args.get('student_lr'),
                            l2_cost=hp_args.get('student_l2_cost')
                            )


        # Create test data loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        test_group_loader = DataLoader(test_group_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        group_info = test_dataset.get_group_info()

        # Obtain results
        score = evaluate(model, test_loader)
        worst_group_score = evaluate_worst_group_accuracy(model, test_group_loader, group_info)
        
        # Save results to dataframe
        new_row = {}
        new_row['training_set'] = random_seed
        new_row['score'] = score
        new_row['worst_group_score'] = worst_group_score

        if model_name == 'l2' or model_name == 'gdro':
            new_row['lr'] = hp_args.get('lr')
            new_row['l2_cost'] = hp_args.get('l2_cost')

        elif model_name == 'mmd':
            new_row['lr'] = hp_args.get('lr')
            new_row['sigma'] = hp_args.get('sigma')
            new_row['alpha'] = hp_args.get('alpha')

        elif model_name == 'tipmi' or model_name == 'tipmi-cf' or model_name == 'ban':
            new_row['student_lr'] = hp_args.get('student_lr')
            new_row['student_l2_cost'] = hp_args.get('student_l2_cost')

        results_df.loc[len(results_df)] = new_row
        print(results_df)



# Process results
file_name = model_name + '_' + str(time.time())
file_name_raw = 'raw_' + file_name

results_df.to_csv(os.path.join(CROSS_VAL_RESULTS_DIRECTORY, file_name_raw), index=False)
if model_name == 'gdro' or model_name == 'mmd':
    results_df_max = results_df.sort_values('worst_group_score').groupby(['training_set']).tail(1).sort_values('training_set')
else:
    results_df_max = results_df.sort_values('score').groupby(['training_set']).tail(1).sort_values('training_set')
results_df_max.to_csv(os.path.join(CROSS_VAL_RESULTS_DIRECTORY, file_name), index=False)

print("Results: ")
print(results_df)
print("Max Results: ")
print(results_df_max)