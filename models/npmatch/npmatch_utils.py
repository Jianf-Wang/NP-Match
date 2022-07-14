import torch
import math
import torch.nn.functional as F
import numpy as np

from train_utils import  ce_loss_np


class Get_Scalar():
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def selecting_with_thresh(logits, p_cutoff, uncertainty_cutoff, class_acc, prob_list):
    pseudo_label = torch.softmax(logits, dim=-1)
    pseudo_label = pseudo_label.mean(0)

    uncertainty = -1.0 *torch.sum(pseudo_label * pseudo_label.log(), dim=-1)

   
    pseudo_label = pseudo_label / pseudo_label.mean(0)
    pseudo_label = pseudo_label / pseudo_label.sum(dim=1, keepdim=True)
    

    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

    # follow FlexMatch
    mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx])))  # convex 
    mask_uncertain = uncertainty.le(uncertainty_cutoff)
    mask = torch.logical_and(mask, mask_uncertain)

    select = max_probs.ge(p_cutoff)
    select_uncertain =  uncertainty.le(uncertainty_cutoff)
    select = torch.logical_and(select, select_uncertain).long()

    return mask.float().mean(), mask, select, max_idx.long(), prob_list


def calculate_loss(logit_context, logit_target, mean_context, sigma_context, mean_target, sigma_target,  sample_T, labels_target_onehot):
    log_var_context = 2 * (sigma_context.log())
    log_var_target = 2 * (sigma_target.log())

    context_B = logit_context.size(1)
    target_B = logit_target.size(1) 
    logit_target_pred = F.softmax(logit_target, dim = -1)
    logit_context_pred = F.softmax(logit_context, dim = -1)

    uncertainty_context_avg = (-1.0 * torch.sum(logit_context_pred.mean(0) * logit_context_pred.mean(0).log())/context_B).detach()
    uncertainty_target_avg = (-1.0 * torch.sum(logit_target_pred.mean(0) * logit_target_pred.mean(0).log())/target_B).detach()
            
    alpha = uncertainty_context_avg/(uncertainty_context_avg + uncertainty_target_avg)
    alpha_var = ((1 - alpha) * (-1 * log_var_context).exp() + alpha * (-1 * log_var_target).exp())**(-1)
    alpha_mean = alpha_var * ((1 - alpha) * (-1 * log_var_context).exp() * mean_context + alpha * (-1 * log_var_target).exp() * mean_target)

    
    skew_uncertain_loss = torch.sum((((1 - alpha) * log_var_context.exp() + alpha * log_var_target.exp()) * (alpha_var ** (-1)) + (alpha_var.log() - (1-alpha) * log_var_context - alpha  * log_var_target) + \
                     (1-alpha) * ((alpha_mean - mean_context)**2) * (alpha_var**(-1)) + alpha * ((alpha_mean - mean_target)**2) * (alpha_var**(-1)) - 1) * 0.5 ) 

 
    unsup_loss = ce_loss_np(logit_target, labels_target_onehot, sample_T) 

    return unsup_loss.mean(), skew_uncertain_loss 


