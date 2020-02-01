import os
import torch
import random
import logging
import numpy as np
from collections import defaultdict
from torch.utils import data
from transformers import BertTokenizer


class Dataset(data.Dataset):
    def __init__(self, data_list, data_ids, tokenizer, label_map, max_len):
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.data_ids = data_ids
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        text, label = self.data_list[self.data_ids[idx]]

        return self._get_feature(text, label)

    def _get_feature(self, text, label):
        word_tokens = ['[CLS]']
        label_list = ['[CLS]']
        label_mask = [0]  # value in (0, 1) - 0 signifies invalid token

        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [self.label_map['[CLS]']]

        # iterate over individual tokens and their labels
        for word, label in zip(text, label):
            tokenized_word = self.tokenizer.tokenize(word)

            for token in tokenized_word:
                word_tokens.append(token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

            label_list.append(label)
            label_ids.append(self.label_map[label])
            label_mask.append(1)
            # len(tokenized_word) > 1 only if it splits word in between, in which case
            # the first token gets assigned NER tag and the remaining ones get assigned
            # X
            for i in range(1, len(tokenized_word)):
                label_list.append('X')
                label_ids.append(self.label_map['X'])
                label_mask.append(0)

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask)

        if len(word_tokens) >= self.max_len:
            word_tokens = word_tokens[:(self.max_len - 1)]
            label_list = label_list[:(self.max_len - 1)]
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]
            label_mask = label_mask[:(self.max_len - 1)]

        assert len(word_tokens) < self.max_len, len(word_tokens)

        word_tokens.append('[SEP]')
        label_list.append('[SEP]')
        input_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        label_ids.append(self.label_map['[SEP]'])
        label_mask.append(0)

        assert len(word_tokens) == len(label_list) == len(input_ids) == len(label_ids) == len(
            label_mask)

        sentence_id = [0 for _ in input_ids]
        attention_mask = [1 for _ in input_ids]

        while len(input_ids) < self.max_len:
            input_ids.append(0)
            label_ids.append(self.label_map['X'])
            attention_mask.append(0)
            sentence_id.append(0)
            label_mask.append(0)

        assert len(word_tokens) == len(label_list)
        assert len(input_ids) == len(label_ids) == len(attention_mask) == len(sentence_id) == len(
            label_mask) == self.max_len, len(input_ids)

        input_ids, label_ids, label_mask = torch.LongTensor(input_ids), torch.LongTensor(label_ids), torch.BoolTensor(label_mask)
        attention_mask, sentence_id = torch.LongTensor(attention_mask), torch.LongTensor(sentence_id)

        return [input_ids, label_ids, attention_mask, sentence_id, label_mask]


class DataLoader:
    def __init__(self, data_dir, bert_model_dir, params):
        self.data_dir = data_dir
        self.params = params
        self.tags = self._load_tags() + ['[CLS]', '[SEP]', 'X']

        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=False)

        self._load_sentence_and_tag()

    def _load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def _load_sentence_and_tag(self):
        data = defaultdict(list)

        # Read all files
        for data_type in ['train', 'val', 'test']:
            sentences = [line.strip() for line in open(os.path.join(self.data_dir, data_type, 'sentences.txt'), 'r')]
            tags = [line.strip() for line in open(os.path.join(self.data_dir, data_type, 'tags.txt'), 'r')]
            assert len(sentences) == len(tags)

            for sentence, tag in zip(sentences, tags):
                sentence, tag = sentence.split(), tag.split()
                assert len(sentence) == len(tag)
                data[data_type].append((sentence, tag))
                
        # Generate initialized train set and unlabeled set
        train_size = int(self.params.train_size * len(data['train']))

        self.data_list = data['train'] + data['val'] + data['test']
        unlabeled_ids = np.arange(len(data['train']))
        val_ids = len(unlabeled_ids) + np.arange(len(data['val']))
        test_ids = len(unlabeled_ids) + len(val_ids) + np.arange(len(data['test']))
        train_ids = np.random.choice(unlabeled_ids, train_size)

        self.datasets = {
            'train': Dataset(data_list, train_ids, self.tokenizer, self.tag2idx, self.params.max_len),
            'val': Dataset(data_list, val_ids, self.tokenizer, self.tag2idx, self.params.max_len),
            'test': Dataset(data_list, test_ids, self.tokenizer, self.tag2idx, self.params.max_len),
            'unlabeled': Dataset(data_list, unlabeled_ids, self.tokenizer, self.tag2idx, self.params.max_len)
        }

        logging.info('Dataset Info: train={}, val={}, test={}, unlabeled={}'.format(
            len(train_ids), len(val_ids), len(test_ids), len(unlabeled_ids)
        ))

    def data_iterator(self, data_type, shuffle=False):
        return data.DataLoader(
            dataset=self.datasets[data_type],
            batch_size=self.params.batch_size,
            shuffle=shuffle,
            num_workers=self.params.num_workers
        )

    def update_train(self, indices):
        """Put data from unlabeled set to trani set"""

        sample_data_ids = self.datasets['unlabeled'].data_ids[indices]
        self.datasets['train'].data_ids = np.concatenate((self.datasets['train'].data_ids, sample_data_ids), axis=0)

    def update_unlabeled(self, indices):

        self.datasets['unlabeled'].data_ids = np.delete(self.datasets['unlabeled'].data_ids, indices)

    def unlabled_length(self):

        return len(self.datasets['unlabeled'].data_ids)

    def train_length(self):

        return len(self.datasets['train'].data_ids)
