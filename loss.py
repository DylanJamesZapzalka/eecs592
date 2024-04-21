###########################################################################
#
# Much of this code was taken and edited from the following GitHub repo:
# https://github.com/kohpangwei/group_DRO/tree/master
#
###########################################################################
import torch
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class LossComputer:
    '''
    Used for calcuting the GDRO loss. Much of this code was taken and edited from
    the following GitHub repo:
    https://github.com/kohpangwei/group_DRO/tree/master
    '''
    
    def __init__(self, criterion, is_robust, dataset_info, alpha=0.2, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False):

        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = dataset_info['n_groups']
        self.group_counts = dataset_info['group_counts'].cuda()

        self.group_frac = self.group_counts/self.group_counts.sum()

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        predictions = torch.where(yhat >= 0.5, 1, 0)
        group_acc, group_count = self.compute_group_avg((predictions==y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()



# kernel width using median trick
def set_width_median(x, y=None):
    if y == None:
        x = x.detach().cpu().numpy()
        dists = pdist(x, 'euclidean')
        median_dist = np.median(dists[dists > 0])
    else:
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        dists = cdist(x, y, 'euclidean')
        median_dist = np.median(dists[dists > 0])

    width = np.sqrt(2.) * median_dist
    gamma = - 0.5 / (width ** 2)
    return gamma



def rbf_kernel(x, y=None, mmd_sigma='median'):
    if y is not None:
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        kernel_matrix = torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0)
    else:
        norm = torch.sum(x**2, axis=-1)
        kernel_matrix = norm[:, None] + norm[None, :] - 2 * torch.matmul(x, torch.transpose(x, dim0=1, dim1=0))
    set_width_median(x, y)
    if mmd_sigma == 'median':
        gamma = set_width_median(x, y)
    else:
        gamma = - 0.5 / (mmd_sigma ** 2)
    kernel_matrix = torch.exp(gamma * kernel_matrix)

    return kernel_matrix



class Weighted_MMD(torch.nn.Module):
    def __init__(self, mmd_sigma):
        super(Weighted_MMD, self).__init__()
        self.mmd_sigma = mmd_sigma

    def loss_weights(self, labels, auxiliary_labels):
        label_size = labels.size()[0]
        
        labels = torch.squeeze(labels)

        labels = labels.to(dtype=torch.float32)
        auxiliary_labels = auxiliary_labels.to(dtype=torch.float32)

        p_cancer_white = torch.dot(labels, auxiliary_labels) / label_size
        p_cancer_black = torch.dot(labels, 1 - auxiliary_labels) / label_size
        p_benign_white = torch.dot(1 - labels, auxiliary_labels) / label_size
        p_benign_black = torch.dot(1 - labels, 1 - auxiliary_labels) / label_size

        p_cancer = torch.sum(labels) / label_size
        p_benign= torch.sum(1 - labels) / label_size

        p_white = torch.sum(auxiliary_labels) / label_size
        p_black = torch.sum(1 - auxiliary_labels) / label_size

        cancer_white_weight = p_cancer * p_white / p_cancer_white if p_cancer_white != 0 else 0
        cancer_black_weight = p_cancer * p_black / p_cancer_black if p_cancer_black != 0 else 0
        benign_white_weight = p_benign * p_white / p_benign_white if p_benign_white != 0 else 0
        benign_black_weight = p_benign * p_black / p_benign_black if p_benign_black != 0 else 0

        cancer_white_vector = cancer_white_weight * labels * auxiliary_labels
        cancer_black_vector = cancer_black_weight * labels * (1 - auxiliary_labels)
        benign_white_vector = benign_white_weight * (1 - labels) * auxiliary_labels
        benign_black_vector = benign_black_weight * (1 - labels) * (1 - auxiliary_labels)

        weight_vector = cancer_white_vector + cancer_black_vector + benign_white_vector + benign_black_vector
        weight_vector = weight_vector / torch.sum(weight_vector)

        return weight_vector


    def mmd(self, features, auxiliary_labels, rand_perm=False):
        features = features # Features are wrapped inside of a size 1 tuple
        auxiliary_labels = torch.unsqueeze(auxiliary_labels, dim=1).to(dtype=torch.float32) # Shape = (num_labels, 1)

        kernel_matrix = rbf_kernel(features, mmd_sigma=self.mmd_sigma)
        mask_pos = torch.matmul(auxiliary_labels, torch.transpose(auxiliary_labels, dim0=1, dim1=0))
        mask_neg = torch.matmul(1 - auxiliary_labels, torch.transpose(1 - auxiliary_labels, dim0=1, dim1=0))
        mask_pos_neg = torch.matmul(auxiliary_labels, torch.transpose(1 - auxiliary_labels, dim0=1, dim1=0))

        weight_pos = 1 / torch.sum(mask_pos) if torch.sum(mask_pos) != 0 else 0
        weight_neg = 1 / torch.sum(mask_neg) if torch.sum(mask_neg) != 0 else 0
        weight_pos_neg = 1 / torch.sum(mask_pos_neg) if torch.sum(mask_pos_neg) != 0 else 0

        mean_pos = weight_pos * torch.sum(kernel_matrix * mask_pos)
        mean_neg = weight_neg * torch.sum(kernel_matrix * mask_neg)
        mean_pos_neg = weight_pos_neg * torch.sum(kernel_matrix * mask_pos_neg)

        mmd = mean_pos + mean_neg - 2 * mean_pos_neg
        mmd = torch.max(torch.tensor(0), mmd)

        return mmd



    def forward(self, yhat, y, features, auxiliary_labels):
        # Calculate cross entropy loss
        ce_loss_function = torch.nn.BCELoss(reduction='none')
        ce_loss = ce_loss_function(yhat, y)

        # Calculate weighted cross entropy loss
        weight_vector = self.loss_weights(y, auxiliary_labels)
        ce_loss = torch.squeeze(ce_loss)
        weighted_ce_loss = torch.dot(ce_loss, weight_vector)

        # Calculate the mmd loss
        mmd_loss = self.mmd(features, auxiliary_labels)

        return mmd_loss, weighted_ce_loss