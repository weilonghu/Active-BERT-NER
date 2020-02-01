import torch
import numpy as np
from torch.distributions.categorical import Categorical


class ActiveStrategy(object):

    def __init__(self, num_labels, uncertainty_strategy, density_strategy):

        self.num_labels = num_labels

        if uncertainty_strategy == 'none' and density_strategy == 'none':
            raise ValueError('Uncertainty and density strategies can not be none at the same time')

        if uncertainty_strategy != 'none':
            self.uncertainty_strategy = uncertainty_strategy.strip().split('+')
        else:
            self.uncertainty_strategy = []

        if density_strategy != 'none':
            self.density_strategy = density_strategy.strip().split('+')
        else:
            self.density_strategy = []

        self.label_map = {
            'random_select': 'R',
            'least_confidence': 'LC',
            'token_entropy': 'TE',
            'representative': 'Rep'
        }

    def get_strategy_label(self, use_crf):

        uncertainty_labels = [self.label_map[name] for name in self.uncertainty_strategy]
        density_labels = [self.label_map[name] for name in self.density_strategy]
        uncertainty_labels = '+'.join(uncertainty_labels)
        density_labels = '+'.join(density_labels)

        if len(uncertainty_labels) > 0 and len(density_labels) > 0:
            label = '{}+{}'.format(uncertainty_labels, density_labels)
        elif len(uncertainty_labels) == 0 and len(density_labels) == 0:
            label = 'R'
        else:
            label = '{}{}'.format(uncertainty_labels, density_labels)

        if use_crf is True:
            label = '{}+C'.format(label)

        return label

    def sample_batch(self, total_num, query_num, **kwargs):

        uncertianty_scores = []
        density_scores = []

        for strategy in self.uncertainty_strategy:
            strategy_func = getattr(self, '{}_sample'.format(strategy))
            scores = strategy_func(query_num, **kwargs)
            uncertianty_scores.append(scores)
        for strategy in self.density_strategy:
            strategy_func = getattr(self, '{}_sample'.format(strategy))
            scores = strategy_func(query_num, **kwargs)
            density_scores.append(scores)

        # Just make production
        uncertianty_scores = torch.stack(uncertianty_scores) if len(uncertianty_scores) > 0 else torch.ones(total_num)
        density_scores = torch.stack(density_scores) if len(density_scores) > 0 else torch.ones(total_num)

        # Combine two scores
        scores = torch.prod(torch.cat((uncertianty_scores.view(1, -1), density_scores.view(1, -1)), dim=0), dim=0)

        # Select topk as queried items
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
        scores = np.ones(num_unlabeled, dtype=np.float32)
        scores[indices] += 1

        return torch.from_numpy(scores)

    def least_confidence_sample(self, query_num, confidences, **kwargs):

        confidences = torch.cat(confidences, dim=0)
        scores = 1 - confidences

        return scores

    def pcosine_similarity(self, x1, x2=None, eps=1e-8):
        """Pairwise cosine similary, S(a, b) = 1 - D(a, b)"""

        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    def representative_sample(self, query_num, similarities, **kwargs):

        avg_sims = torch.mean(similarities, dim=1)

        return avg_sims
