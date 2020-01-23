from uncertainty_sampling import token_entropy
from uncertainty_sampling import random_select


class Strategy(object):

    def __init__(self):
        self.register_strategy = {
            'token_entropy': token_entropy,
            'random_select': random_select
        }

    def get_strategy_proxy(self, strategy_name):
        strategy_func = self.register_strategy.get(strategy_name, None)
        if strategy_func is None:
            raise ValueError('Unknown strategy name')

        return strategy_func
