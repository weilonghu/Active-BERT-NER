import torch
import numpy as np
from torch.distributions.categorical import Categorical


class ActiveStrategy(object):

    def __init__(self, num_labels):

        self.num_labels = num_labels
        self.support_strategies = ['random_select', 'token_entropy', 'least_confidence']

    def sample_batch(self, strategy_name, query_num, **kwargs):

        if strategy_name not in self.support_strategies:
            raise ValueError('Unknown stratege name')

        strategy_func = getattr(self, '{}_sample'.format(strategy_name))
        scores = strategy_func(query_num, **kwargs)

        _, indices = torch.topk(scores, query_num)

        return indices

    def token_entropy_sample(self, query_num, logitss, masks, **kwargs):

        token_entropy = []
        for logits, mask in zip(logitss, masks):
            origin_num = logits.size(0)
            categorical = Categorical(logits=logits.view(-1, self.num_labels))
            entropy = categorical.entropy() * mask.view(-1)
            entropy = torch.sum(entropy.view(origin_num, -1), dim=-1)
            lens = torch.sum(mask, dim=-1)
            entropy = entropy / lens
            token_entropy.append(entropy)
        token_entropy = torch.cat(token_entropy, dim=0)

        return token_entropy


    def random_select_sample(self, query_num, num_unlabeled, **kwargs):

        indices = np.random.choice(np.arange(num_unlabeled), query_num)
        scores = np.ones(num_unlabeled)
        scores[indices] += 1

        return torch.from_numpy(scores)

    def least_confidence_sample(self, query_num, confidences, **kwargs):

        confidences = torch.cat(confidences, dim=0)
        scores = 1 - confidences

        return scores
