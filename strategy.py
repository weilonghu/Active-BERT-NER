import torch
import numpy as np
from torch.distributions.categorical import Categorical


class ActiveStrategy(object):

    def __init__(
        self, num_labels, uncertainty_strategy, density_strategy,
        confidence_threshold, size_threshold):

        self.num_labels = num_labels
        self.beta = 1
        self.size_threshold = size_threshold

        if confidence_threshold.startswith('top') or confidence_threshold.startswith('abs'):
            self.confidence_threshold = confidence_threshold
        else:
            raise ValueError('confidence_threshold format error')

        self.label2func = {
            'r': 'random_select',
            'lc': 'least_confidence',
            'te': 'token_entropy',
            'tte': 'total_token_entropy',
            'du': 'density_unlabeled',
            'dl': 'density_labeled'
        }

        if uncertainty_strategy == 'none' and density_strategy == 'none':
            raise ValueError('Uncertainty and density strategies can not be none at the same time')

        if uncertainty_strategy != 'none':
            self.uncertainty_strategy = [self.label2func[x] for x in uncertainty_strategy.strip().split('+')]
        else:
            self.uncertainty_strategy = []

        if density_strategy != 'none':
            self.density_strategy = [self.label2func[x] for x in density_strategy.strip().split('+')]
        else:
            self.density_strategy = []

    def output_label(self):

        return 'dl' in self.density_strategy

    def get_strategy_label(self, use_crf):

        uncertainty_labels = '+'.join(self.uncertainty_strategy)
        density_labels = '+'.join(self.density_strategy)

        if len(uncertainty_labels) > 0 and len(density_labels) > 0:
            label = '{}+{}'.format(uncertainty_labels, density_labels)
        elif len(uncertainty_labels) == 0 and len(density_labels) == 0:
            raise ValueError('Uncertainty and density strategies can not be none at the same time')
        else:
            label = '{}{}'.format(uncertainty_labels, density_labels)

        if use_crf is True:
            label = '{}+C'.format(label)

        return label

    def sample_batch(self, labeled_num, unlabeled_num, query_num, **kwargs):
        """Qeury a batch for active learning. Before active learning, peform self-training

        Args:
            labeled_num: (int) the size of train set
            unlabeled_num: (int) the size of unlabeled data set
            query_num: (int) query batch size
        Return:
            indices: (tuple) indices of machine labeled examples and human labeled examples
        """
        if labeled_num >= self.size_threshold:
            machine_indices = self.bootstrapping(query_num, **kwargs)
            machine_indices = machine_indices.cpu()
        else:
            machine_indices = None

        uncertianty_scores = []
        density_scores = []

        for strategy in self.uncertainty_strategy:
            strategy_func = getattr(self, '{}_sample'.format(strategy))
            scores = strategy_func(query_num, **kwargs)
            uncertianty_scores.append(scores.cpu())
        for strategy in self.density_strategy:
            strategy_func = getattr(self, '{}_sample'.format(strategy))
            scores = strategy_func(query_num, **kwargs)
            density_scores.append(scores.cpu())

        # Just make production
        uncertianty_scores = torch.stack(uncertianty_scores) if len(uncertianty_scores) > 0 else torch.ones(unlabeled_num)
        density_scores = torch.stack(density_scores) if len(density_scores) > 0 else torch.ones(unlabeled_num)

        # Combine two scores
        scores = torch.prod(torch.cat((uncertianty_scores.view(1, -1), density_scores.view(1, -1)), dim=0), dim=0)

        # Select topk as queried items
        _, human_indices = torch.topk(scores, query_num)

        return (machine_indices, human_indices)

    def _token_entropy(self, query_num, logitss, masks, norm, **kwargs):

        token_entropy = []
        for logits, mask in zip(logitss, masks):
            origin_num = logits.size(0)
            categorical = Categorical(logits=logits.view(-1, self.num_labels))
            entropy = categorical.entropy() * mask.view(-1)
            entropy = torch.sum(entropy.view(origin_num, -1), dim=-1)
            if norm is True:
                lens = torch.sum(mask, dim=-1)
                entropy = entropy / lens
            token_entropy.append(entropy)
        token_entropy = torch.cat(token_entropy, dim=0)


        return token_entropy

    def token_entropy_sample(self, query_num, **kwargs):

        return self._token_entropy(query_num=query_num, norm=True, **kwargs)

    def total_token_entropy_sample(self, query_num, **kwargs):

        return self._token_entropy(query_num=query_num, norm=False, **kwargs)

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

    def density_unlabeled_sample(self, query_num, unlabel_sims, **kwargs):

        unlabel_sims = torch.mean(unlabel_sims, dim=1)
        scores = torch.pow(unlabel_sims, self.beta)

        return scores

    def density_labeled_sample(self, query_num, label_sims, **kwargs):

        label_sims = torch.mean(label_sims, dim=1)
        scores = scores * torch.exp(-1 * label_sims)

        return scores

    def bootstrapping(self, query_num, confidences, **kwargs):

        confidences = torch.cat(confidences, dim=0)
        threshold = float(self.confidence_threshold.split('_')[1])

        if self.confidence_threshold.startswith('abs') and threshold <= 1:
            indices = torch.nonzero((confidences > threshold).to(torch.int))
            return indices.view(-1)
        elif self.confidence_threshold.startswith('top'):
            if threshold < 1:
                _, indices = torch.topk(confidences, int(confidences.size(0) * threshold))
            else:
                _, indices = torch.topk(confidences, int(threshold))
            return indices.view(-1)
        else:
            raise ValueError('Confidence threshold format error')
