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



class Weighted_MMD(torch.nn.Module):
    '''
    Used for calcuting the Weigted MMD loss. The paper describing this
    loss function can be found here:
    https://proceedings.mlr.press/v151/makar22a/makar22a.pdf
    '''

    def __init__(self, mmd_sigma):
        super(Weighted_MMD, self).__init__()
        self.mmd_sigma = mmd_sigma


    def rbf_kernel(self, x, y=None):
        if y is not None:
            x_norm = (x ** 2).sum(1).view(-1, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
            kernel_matrix = torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0)
        else:
            norm = torch.sum(x**2, axis=-1)
            kernel_matrix = norm[:, None] + norm[None, :] - 2 * torch.matmul(x, torch.transpose(x, dim0=1, dim1=0))

        gamma = - 0.5 / (self.mmd_sigma ** 2)
        kernel_matrix = torch.exp(gamma * kernel_matrix)

        return kernel_matrix


    def loss_weights(self, labels, auxiliary_labels):

        pos_label_pos_aux_weight = torch.sum(labels * auxiliary_labels) / torch.sum(auxiliary_labels)
        neg_label_pos_aux_weight = torch.sum((1.0 - labels) * auxiliary_labels) / torch.sum(auxiliary_labels)

        pos_label_neg_aux_weight = torch.sum(labels * (1.0 - auxiliary_labels)) / torch.sum((1.0 - auxiliary_labels))
        neg_label_neg_aux_weight = torch.sum((1.0 - labels) * (1.0 - auxiliary_labels)) / torch.sum((1.0 - auxiliary_labels))

        # Positive weights
        weights_pos = labels * pos_label_pos_aux_weight + (1.0 - labels) * neg_label_pos_aux_weight
        weights_pos = 1 / weights_pos
        weights_pos = auxiliary_labels * weights_pos
        weights_pos = torch.mean(labels) * labels * weights_pos + \
            torch.mean(1 - labels) * (1.0 - labels) * weights_pos
        
        # Negative weights
        weights_neg = labels * pos_label_neg_aux_weight + (1.0 - auxiliary_labels) * neg_label_neg_aux_weight
        weights_neg = 1.0 / weights_neg
        weights_neg = (1.0 - auxiliary_labels) * weights_neg
        weights_neg = torch.mean(labels) * labels * weights_neg + \
            torch.mean(1.0 - labels) * (1.0 - labels) * weights_neg
        
        # Make sure there are no nan errors
        weights_pos = torch.nan_to_num(weights_pos)
        weights_neg = torch.nan_to_num(weights_neg)
        weights = weights_pos + weights_neg

        return weights, weights_pos, weights_neg


    def mmd(self, features, auxiliary_labels, weights_pos, weights_neg):

        kernel_matrix = self.rbf_kernel(features)

        mask_pos = torch.matmul(auxiliary_labels, torch.transpose(auxiliary_labels, dim0=1, dim1=0))
        mask_neg = torch.matmul(1 - auxiliary_labels, torch.transpose(1 - auxiliary_labels, dim0=1, dim1=0))
        mask_pos_neg = torch.matmul(auxiliary_labels, torch.transpose(1 - auxiliary_labels, dim0=1, dim1=0))
        mask_neg_pos = torch.matmul(1 - auxiliary_labels, torch.transpose(auxiliary_labels, dim0=1, dim1=0))

        pos_kernel_mean = kernel_matrix * mask_pos
        pos_kernel_mean = pos_kernel_mean * torch.transpose(weights_pos, dim0=1, dim1=0)
        pos_kernel_mean = torch.divide(torch.sum(pos_kernel_mean, dim=1), torch.sum(weights_pos))
        pos_kernel_mean = torch.divide(torch.sum(pos_kernel_mean * torch.squeeze(weights_pos)), torch.sum(weights_pos))

        neg_kernel_mean = kernel_matrix * mask_neg
        neg_kernel_mean = neg_kernel_mean * torch.transpose(weights_neg, dim0=1, dim1=0)
        neg_kernel_mean = torch.divide(torch.sum(neg_kernel_mean, dim=1), torch.sum(weights_neg))
        neg_kernel_mean = torch.divide(torch.sum(neg_kernel_mean * torch.squeeze(weights_neg)), torch.sum(weights_neg))

        neg_pos_kernel_mean = kernel_matrix * mask_neg_pos
        neg_pos_kernel_mean = neg_pos_kernel_mean * torch.transpose(weights_pos, dim0=1, dim1=0)
        neg_pos_kernel_mean = torch.divide(torch.sum(neg_pos_kernel_mean, dim=1), torch.sum(weights_pos))
        neg_pos_kernel_mean = torch.divide(torch.sum(neg_pos_kernel_mean * torch.squeeze(weights_neg)), torch.sum(weights_neg))

        pos_neg_kernel_mean = kernel_matrix * mask_pos_neg
        pos_neg_kernel_mean = pos_neg_kernel_mean * torch.transpose(weights_neg, dim0=1, dim1=0)
        pos_neg_kernel_mean = torch.divide(torch.sum(pos_neg_kernel_mean, dim=1), torch.sum(weights_neg))
        pos_neg_kernel_mean = torch.divide(torch.sum(pos_neg_kernel_mean * torch.squeeze(weights_pos)), torch.sum(weights_pos))

        mmd_loss = pos_kernel_mean + neg_kernel_mean - (pos_neg_kernel_mean + neg_pos_kernel_mean)
        mmd_loss = torch.max(torch.tensor(0), mmd_loss)

        return mmd_loss


    def forward(self, yhat, y, features, z):
        # Get weights and auxiliary labels
        auxiliary_labels = z[:, 0]
        weights = z[:, 1]
        weights_pos = z[:, 2]
        weights_neg = z[:, 3]

        # Calculate cross entropy loss
        ce_loss_function = torch.nn.BCELoss(reduction='none')
        ce_loss = ce_loss_function(yhat, y)

        # Calculate weighted cross entropy loss
        weighted_ce_loss = weights * ce_loss
        weighted_ce_loss = torch.sum(weighted_ce_loss) / torch.sum(weights)

        # Calculate the mmd loss
        auxiliary_labels = torch.unsqueeze(auxiliary_labels, dim=1)
        weights_pos = torch.unsqueeze(weights_pos, dim=1)
        weights_neg = torch.unsqueeze(weights_neg, dim=1)
        mmd_loss = self.mmd(features, auxiliary_labels, weights_pos, weights_neg)
        mmd_loss = 0

        return mmd_loss, weighted_ce_loss