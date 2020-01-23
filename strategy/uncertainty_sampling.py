"""Uncertainty sampling"""
import torch
import random


def token_entropy(scores, mask, num):
    """Total token entropy"""
    scores = scores * mask
    entroy = torch.sum(torch.sum(scores * torch.log(scores), dim=-1), dim=-1)
    lens = torch.sum(mask, dim=-1)
    token_entropy = -entroy / lens
    _, indices = torch.topk(token_entropy, num)

    return indices


def random_select(scores, mask, num):
    """Random select"""
    indices = random.choices(list(range(scores.size(0))), num)

    return indices


def least_confidence():
    pass
