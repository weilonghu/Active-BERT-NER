import os
import torch
import random
from collections import defaultdict
from torch.utils import data
from transformers import BertTokenizer


class Dataset(data.Dataset):
    def __init__(self, data_list, tokenizer, label_map, max_len):
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        text, label = self.data_list[idx]

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

        return input_ids, label_ids, attention_mask, sentence_id, label_mask


class DataLoader:
    def __init__(self, data_dir, bert_model_dir, params):
        self.data_dir = data_dir
        self.params = params
        self.tags = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]", "X"]

        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=False)

        self._load_data()

    def _load_data(self):
        data = defaultdict(list)

        # Read all files
        for data_type in ['train', 'val', 'test']:
            with open(os.path.join(self.data_dir, data_type + '.txt'), 'r') as f:
                sentence = []
                label = []
                for line in f:
                    if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                        if len(sentence) > 0:
                            data[data_type].append((sentence, label))
                            sentence = []
                            label = []
                        continue
                    splits = line.split(' ')
                    sentence.append(splits[0])
                    label.append(splits[-1][:-1])

                if len(sentence) > 0:
                    data[data_type].append((sentence, label))
                    sentence = []
                    label = []
        # Generate initialized train set and unlabeled set
        train_size = int(self.params.train_size * len(data['train']))
        train_data = random.choices(data['train'], k=train_size)

        self.datasets = {
            'train': Dataset(train_data, self.tokenizer, self.tag2idx, self.params.max_len),
            'val': Dataset(data['val'], self.tokenizer, self.tag2idx, self.params.max_len),
            'test': Dataset(data['test'], self.tokenizer, self.tag2idx, self.params.max_len),
            'unlabeled': Dataset(data['train'], self.tokenizer, self.tag2idx, self.params.max_len)
        }

    def data_iterator(self, data_type, shuffle=False):
        return data.DataLoader(
            dataset=self.datasets[data_type],
            batch_size=self.params.batch_size,
            shuffle=shuffle
        )

    def update_dataset(self, indices):
        """Put data from unlabeled set to trani set"""

        sample_data = [self.datasets['unlabeled'][i] for i in indices]
        # put into train set
        self.datasets['train'].data_list.extend(sample_data.copy())
        # remove from unlabeled set
        unlabeled_data = [v for i, v in enumerate(self.datasets['unlabeled']) if i not in frozenset(indices)]
        self.datasets['unlabeled'] = unlabeled_data

    def unlabled_length(self):

        return len(self.datasets['unlabeled'].data_list)
