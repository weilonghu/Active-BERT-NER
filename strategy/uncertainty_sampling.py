"""Uncertainty sampling"""
import torch
import random
import numpy as np
from torch.distributions.categorical import Categorical


def token_entropy(scores, num_unlabeled, batch_size, num_classes=12):
    """Total token entropy"""
    original_logits = [x[0] for x in scores]
    original_mask = [(x[1] != -1).float() for x in scores]

    token_entropy = []
    for logits, mask in zip(original_logits, original_mask):
        origin_num = logits.size(0)
        categorical = Categorical(logits=logits.view(-1, num_classes))
        entropy = categorical.entropy() * mask.view(-1)
        entropy = torch.sum(entropy.view(origin_num, -1), dim=-1)
        lens = torch.sum(mask, dim=-1)
        entropy = entropy / lens
        token_entropy.append(entropy)
    token_entropy = torch.cat(token_entropy, dim=0)
    _, indices = torch.topk(token_entropy, batch_size)

    return indices


def random_select(scores, num_unlabeled, batch_size):
    """Random select"""
    indices = np.random.choice(np.arange(num_unlabeled), batch_size)

    return indices


def least_confidence():
    pass
