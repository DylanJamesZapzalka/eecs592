import torch
from .datasets import SkinCancerDataset, TeacherDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import time
import os
import argparse
from .utils import train, evaluate, train_teacher_model
from .utils import get_teacher_logits, evaluate_worst_group_accuracy
from sklearn.model_selection import KFold


# Train on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using the following device: " + str(device))

# Detects any issues with back propogation
torch.autograd.set_detect_anomaly(True)

SKIN_CANCER_DATASET_DIR = '/home/dylanz/eecs592/data/skin_cancer'
EVALUATION_RESULTS_DIRECTORY = '/home/dylanz/eecs592/results/evaluation'
NUM_FOLDS = 5

# Number of workers
NUM_WORKERS = 10

TEST_DISTRIBUTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# Generate command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',
                    type=str,
                    help='Type of model to train -- can be \"l2\", \"tipmi\", \"tipmi-cf\", \"teacher\", \"ban\", \"gdro\", or \"mmd\" ')
parser.add_argument('--num_datasets',
                    type=int,
                    nargs='?',
                    help='Number of datasets to perform evaluation over.')
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
                    nargs='*',
                    default=[],
                    help='Learning rate for the teacher')
parser.add_argument('--teacher_l2_cost',
                    type=float,
                    nargs='*',
                    default=[],
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
teacher_lr = args.teacher_lr
teacher_l2_cost = args.teacher_l2_cost


# Create dataframe to store the results
columns = []
for i in TEST_DISTRIBUTIONS:
    columns.append(str(i))
results_df = pd.DataFrame(columns = columns)

print(num_datasets)
for random_seed in tqdm(range(num_datasets), position=0, desc='Random Seeds'):

    print(random_seed)

    # Get parameters for corresponding dataset
    if model_name == 'l2' or model_name == 'gdro':
        lr = args.lr[random_seed]
        l2_cost = args.l2_cost[random_seed]
    elif model_name == 'tipmi' or model_name == 'tipmi-cf':
        student_lr = args.student_lr[random_seed]
        student_l2_cost = args.student_l2_cost[random_seed]
        teacher_lr = args.student_lr[random_seed]
        teacher_l2_cost = args.student_l2_cost[random_seed]
    elif model_name == 'mmd':
        lr = args.lr[random_seed]
        sigma = args.sigma[random_seed]
        alpha = args.alpha[random_seed]

    # Obtain dataset
    csv_path = os.path.join(SKIN_CANCER_DATASET_DIR, f'training_{random_seed}', 'training.csv')
    full_df = pd.read_csv(csv_path)

    # Get dataset
    include_extra = model_name == 'tipmi' or model_name == 'tipmi-cf' or model_name == 'ban'
    include_group = model_name == 'gdro'
    include_aux = model_name == 'mmd'
    
    train_dataset = SkinCancerDataset(full_df, include_group=include_group, include_aux=include_aux)
    val_dataset = SkinCancerDataset(full_df, include_extra=include_extra)



    # Create data loader for l2 or the teacher
    if model_name == 'l2' or model_name == 'teacher' or model_name == 'gdro' or model_name == 'mmd':

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)


    # Create data loader for tipmi
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

    # Create data loader for ban
    elif model_name == 'ban':

        # Train the teacher model with a temporary data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        teacher_model = train_teacher_model(train_loader, student_lr, student_l2_cost, num_epochs, True)
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
        for fold, (teacher_train_ind, teacher_val_ind) in enumerate(kfold.split(full_df)):

            # Get k-fold training data
            train_k_df = full_df.iloc[teacher_train_ind]
            val_df = full_df.iloc[teacher_val_ind]

            # Get k-fold datasets
            train_k_dataset = SkinCancerDataset(train_k_df)
            val_dataset = SkinCancerDataset(val_df, include_extra = True)


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



    # Train the model
    if model_name == 'teacher':
        model = train_teacher_model(train_loader,
                                    teacher_lr,
                                    teacher_l2_cost,
                                    num_epochs
                                    )
    elif model_name == 'l2':
        model = train(model_name,
                        num_epochs,
                        train_loader,
                        lr=lr,
                        l2_cost=l2_cost
                        )
    elif model_name == 'gdro':
        dataset_info = train_dataset.get_group_info()
        model = train(model_name,
                        num_epochs,
                        train_loader,
                        lr=lr,
                        l2_cost=l2_cost,
                        dataset_info=dataset_info
                        )
    elif model_name == 'mmd':
        mmd_info = {'sigma': sigma, 'alpha': alpha}
        model = train(model_name,
                        num_epochs,
                        train_loader,
                        lr=lr,
                        l2_cost=0,
                        mmd_info=mmd_info
                        )  
    elif model_name == 'tipmi' or model_name == 'tipmi-cf' or model_name == 'ban':
        model = train(model_name,
                        num_epochs,
                        train_loader,
                        lr=student_lr,
                        l2_cost=student_l2_cost
                        )


    # Analyze the model over various distributions
    new_row = {}
    for i in tqdm(range(len(TEST_DISTRIBUTIONS)), position=1, desc='Distributions', leave=False):

        # Obtain dataset that follows the specified distribution
        csv_path = os.path.join(SKIN_CANCER_DATASET_DIR, 'testing', str(TEST_DISTRIBUTIONS[i]), 'testing.csv')
        test_df = pd.read_csv(csv_path)
        test_dataset = SkinCancerDataset(test_df)

        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

        # Obtain results
        score = evaluate(model, test_loader)
        new_row[str(TEST_DISTRIBUTIONS[i])] = score

    results_df.loc[len(results_df)] = new_row
    print(results_df)

# Save results
file_name = os.path.join(EVALUATION_RESULTS_DIRECTORY, model_name + '_' + str(time.time()))
results_df.to_csv(file_name, index=False)
print("Resuts: ")
print(results_df)