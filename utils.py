import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from .models import BinaryPreTrainedNet
from .loss import LossComputer, Weighted_MMD
import numpy as np
import warnings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ignore these warnings that occur when calculating worst group accuracy
warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "invalid value encountered in scalar divide")

def train(
        model_name,
        num_epochs,
        train_loader,
        lr,
        l2_cost,
        dataset_info = None,
        mmd_info = None
        ):
    """
    Used for creating and training a model
    :param model_name: The type of model that is being trained
    :param num_epochs: Number of epochs the model will be trained for
    :param train_loader: The training dataset loader
    :param lr: Learning rate
    :param l2_cost: The L2 regularization cost
    :param dataset_info: Information used for calculating GDRO loss
    :param mmd_info: Info used for calculating Weighted MMD loss
    """

    # Get model
    model = BinaryPreTrainedNet('resnet').to(device)

    # Obtain optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_cost)

    # Used for GDRO
    loss_computer = None
    if model_name == 'gdro':
        loss_function = torch.nn.BCELoss(reduction = 'none')
        loss_computer = LossComputer(loss_function, True, dataset_info, alpha=1)

    # Train the model over n epochs
    for _ in tqdm(range(num_epochs), position=2, leave=False, desc='Epochs'):

        # Iterate through dataset
        for batch in train_loader:
            loss = get_loss(model,
                            model_name,
                            batch,
                            loss_computer=loss_computer,
                            mmd_info=mmd_info
                            )

            # Perform backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model



def get_loss(
        model,
        model_name,
        batch,
        loss_computer,
        mmd_info
        ):
    """
    Used for getting the loss for a single batch
    :param model: The machine learning model being trained
    :param model_name: The type of model that is being trained
    :param batch: A single batch used to calculate the loss
    :param loss_computer: A helper class used for calculating GDRO loss
    :param mmd_info: Info used for calculating Weighted MMD loss
    """

    (x, y, z) = batch

    # Set hooks for MMD loss
    activation = {}
    if model_name == 'mmd':
        def getActivation(name):
            def hook(model, input, output):
                activation[name] = input
            return hook
        handle = model.model.fc.register_forward_hook(getActivation('features'))


    # Obtain loss functions needed for each test
    if model_name == 'tipmi' or model_name == 'tipmi-cf':
        ce_loss_function = torch.nn.MSELoss()
    elif model_name == 'l2':
            ce_loss_function = torch.nn.BCELoss()

    # Load data onto GPU (if it exists)
    x = x.to(device).to(dtype=torch.float32)
    y = y.to(device).to(dtype=torch.float32)
    z = z.to(device).to(dtype=torch.float32)

    # Calculate the loss
    if model_name == 'l2':
        yhat = model(x)
        y = torch.unsqueeze(y, dim=1)
        loss = ce_loss_function(yhat, y)

    elif model_name == 'tipmi' or model_name == 'tipmi-cf':
        student_logits = model.forward_logits(x)
        student_logits = torch.squeeze(student_logits)
        loss = ce_loss_function(student_logits, y)

    elif model_name == 'gdro':
        yhat = model(x)
        y = torch.unsqueeze(y, dim=1)
        loss = loss_computer.loss(yhat, y, group_idx=z, is_training=True)                

    elif model_name == 'mmd':
        yhat = model(x)
        y = torch.unsqueeze(y, dim=1)
        features = activation['features'][0]
        sigma = mmd_info['sigma']
        sigma = torch.tensor(sigma).to(device)
        alpha = mmd_info['alpha']
        alpha = torch.tensor(alpha).to(device)
        mmd_loss_function = Weighted_MMD(sigma)
        mmd_loss, weighted_ce_loss = mmd_loss_function(yhat, y, features, z)
        loss = weighted_ce_loss + alpha * mmd_loss

    return loss



def evaluate(model, data_loader):
    """
    Used for calculating the aucroc of a model
    :param model: The machine learning model being trained
    :param data_loader: The testing dataset loader
    :param dataset: Name of the dataset being used
    """
    with torch.no_grad():
        all_yhat, all_y = torch.Tensor().to(device), torch.Tensor().to(device)

        for x, y, _ in data_loader:

            x = x.to(device)
            y = y.to(device)
            yhat = model(x)

            all_yhat = torch.cat((all_yhat, yhat))
            all_y = torch.cat((all_y, y))

        # Return auroc
        auroc = roc_auc_score(all_y.cpu().numpy(), all_yhat.cpu().numpy())

        return auroc



def evaluate_worst_group_accuracy(model, data_loader, group_info):
    """
    Used for calculating the worst group accuracy of a model
    :param model: The machine learning model being trained
    :param data_loader: The testing dataset loader
    :param group_info: A dictionary containing information of each group in the dataset
    :param dataset: Name of the dataset being used
    """
    with torch.no_grad():
        all_yhat, all_y, all_group = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)
        for x, y, group in data_loader:
            
            group = group.to(device)
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            all_yhat = torch.cat((all_yhat, yhat))
            all_y = torch.cat((all_y, y))
            all_group = torch.cat((all_group, group))

        # Return accuracy
        accuracy_list = []
        for i in range(group_info['n_groups']):
            group_indices = all_group == i
            yhat = all_yhat[group_indices]
            y = all_y[group_indices]

            group = all_group[group_indices]

            fpr, tpr, thresholds = roc_curve(all_y.cpu().numpy(), all_yhat.cpu().numpy())
            j = tpr - fpr
            index = np.argmax(j)
            threshold = thresholds[index]
            labels = torch.where(yhat >= threshold, 1, 0)
            accuracy = torch.sum((labels.squeeze() == y.squeeze()).long()) / group.shape[0]
            accuracy = accuracy.cpu().numpy()

            accuracy_list.append(accuracy)

        worst_group_accuracy = min(accuracy_list)
        return worst_group_accuracy



def evaluate_teacher(model, data_loader):
    """
    Used for calculating the aucroc of a teacher model
    :param model: The machine learning model being trained
    :param data_loader: The testing dataset loader
    :param dataset: Name of the dataset being used
    """
    with torch.no_grad():
        all_yhat, all_y = torch.Tensor().to(device), torch.Tensor().to(device)

        for _, y, x in data_loader:

            x = x.to(device)
            y = y.to(device)
            yhat = model(x)

            all_yhat = torch.cat((all_yhat, yhat))
            all_y = torch.cat((all_y, y))

        # Return auroc
        auroc = roc_auc_score(all_y.cpu().numpy(), all_yhat.cpu().numpy())

        return auroc


def train_teacher_model(train_loader, lr, l2_weight, num_epochs):
    """
    Used for creating and training a teacher model
    :param dataset: Name of the dataset being used
    :param train_loader: The training dataset loader
    :param lr: Learning rate
    :param l2_weight: The L2 regularization cost
    :param num_epochs: Number of epochs the model will be trained for
    :param ban: Indicates if a model is a BAN
    """

    # Load the model
    model = BinaryPreTrainedNet('resnet')
    model.to(device)

    # Obtain optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)
    loss_function = torch.nn.BCELoss()

    for _ in range(num_epochs):

        for batch in train_loader:
            
            (_, y, x) = batch

            x = x.to(device).to(dtype=torch.float32)
            y = y.to(device).to(dtype=torch.float32)
            y = torch.unsqueeze(y, dim=1)
            yhat = model(x)
            loss = loss_function(yhat, y)      

            # Perform backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model



def get_teacher_logits(teacher_model, val_loader):
    """
    Used for getting the logits of a teacher model
    :param teacher_model: The teacher model
    :param val_loader: The validation dataset loader
    :param dataset: The name of the dataset being used
    :param ban: Indicates if a model is a BAN
    """

    with torch.no_grad():

        extra_list = []
        teacher_logits_list = []

        for x, y, z, extra in val_loader:

            # If BAN, train teacher without privileged info
            x = z

            # Get the logits
            x = x.to(device)
            teacher_logits = teacher_model.forward_logits(x)
            teacher_logits = torch.squeeze(teacher_logits)

            # Save info
            if teacher_logits.numel() > 1: # Prevents bug from occurring when final batch size is 1
                extra_list = extra_list + list(extra)
                teacher_logits_list.append(teacher_logits.cpu())
    
    teacher_logits_list = torch.concat(teacher_logits_list, dim=0)
    
    return extra_list, teacher_logits_list