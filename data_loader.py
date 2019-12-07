"""Data loader"""

import random
import numpy as np
import os
import torch
from transformers import BertTokenizer


class DataLoader(object):
    def __init__(self, data_dir, bert_model_dir, params, token_pad_idx=0, tag_pad_idx=None):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.UNIQUE_LABELS = ['[CLS]', '[SEP]', 'X']

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=False)

        self.token_pad_idx = token_pad_idx
        self.tag_pad_idx = tag_pad_idx if tag_pad_idx is not None else self.tag2idx['X']
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        tags.extend(self.UNIQUE_LABELS)
        return tags

    def load_sentences_tags(self, sentences_file, tags_file, d):
        """Loads sentences and tags from their corresponding files.
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []
        tags = []

        with open(sentences_file, 'r') as file:
            for line in file:
                # replace each token by its index
                tokens = line.strip().split(' ')
                subwords = list(map(self.tokenizer.tokenize, tokens))
                subword_lengths = list(map(len, subwords))
                subwords = ['CLS'] + [item for indices in subwords for item in indices]
                token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
                sentences.append((self.tokenizer.convert_tokens_to_ids(subwords), token_start_idxs))

        with open(tags_file, 'r') as file:
            for line in file:
                # replace each tag by its index
                tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                tags.append(tag_seq)

        # checks to ensure there is a tag for each token
        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
            assert len(tags[i]) == len(sentences[i][-1])

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['tags'] = tags
        d['size'] = len(sentences)

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}

        if data_type in ['train', 'val', 'test']:
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")
        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_token_starts: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        # one pass over data
        for i in range(data['size'] // self.batch_size):
            # fetch sentences and tags
            sentences = [data['data'][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
            tags = [data['tags'][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]

            # generate 'x' labels, then input_ids and tags have equal length respectively.
            xlables = []
            for idx in range(len(sentences)):
                subwords, token_start_idxs = sentences[idx]
                xlabel = [self.tag2idx.get('[CLS]')] + [self.tag2idx.get('X')] * (len(subwords) - 1)
                for i, start in enumerate(token_start_idxs):
                    xlabel[start] = tags[idx][i]
                xlables.append(xlabel)
            tags = xlables

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_subwords_len = max([len(s[0]) for s in sentences])
            max_subwords_len = min(batch_max_subwords_len + 1, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_subwords_len))
            batch_tags = self.tag_pad_idx * np.ones((batch_len, max_subwords_len))
            batch_token_starts = []

            # copy the data to the numpy array
            # the last subword must be '[SEP]'
            for j in range(batch_len):
                cur_subwords_len = len(sentences[j][0])
                if cur_subwords_len < max_subwords_len:
                    batch_data[j][:cur_subwords_len] = sentences[j][0]
                    batch_data[j][cur_subwords_len] = self.sep_token_id
                    batch_tags[j][:cur_subwords_len] = tags[j]
                    batch_tags[j][cur_subwords_len] = self.tag2idx.get('[SEP]')
                else:
                    batch_data[j][:max_subwords_len - 1] = sentences[j][0][:max_subwords_len - 1]
                    batch_data[j][max_subwords_len - 1] = self.sep_token_id
                    batch_tags[j][:max_subwords_len - 1] = tags[j][:max_subwords_len - 1]
                    batch_tags[j][max_subwords_len - 1] = self.tag2idx.get('[SEP]')

                token_start_idx = sentences[j][-1]
                token_starts = np.zeros(max_subwords_len)
                token_starts[[idx for idx in token_start_idx if idx < max_subwords_len - 1]] = 1
                batch_token_starts.append(token_starts)

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_token_starts = torch.tensor(batch_token_starts, dtype=torch.bool)
            batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_token_starts, batch_tags = batch_data.to(self.device), batch_token_starts.to(self.device), batch_tags.to(self.device)

            yield batch_data, batch_token_starts, batch_tags
