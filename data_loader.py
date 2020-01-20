import os
import torch
from collections import defaultdict
from torch.utils import data


class LDataset(data.Dataset):
    """Labeled dataset"""

    def __init__(self, data_list, tokenizer, label_map, max_len):
        """ Construct a dataset for training/evaluating.

        Args:
            data_list: (list) [(text, label)]
            tokenizer: (BertTokenizer) tokenize words
            label_map: (dict) convert tags to ids
            max_len: (int) maximum length of sequences
        """
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """transform an example in dataset

        Return:
            output: (list) [input_ids, label_ids, label_mask, sentence_id, attention_mask]
        """

        text, label = self.data_list[idx]

        return self._generate_feature(text, label)

    def _generate_feature(self, text, label):

        # the first token must be '[CLS]' and the first label must be '[CLS]' too.
        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [self.label_map['[CLS]']]
        label_mask = [0]

        # iterate over individual tokens and their labels
        for word, label in zip(text.split(), label):
            tokenized_word = self.tokenizer.tokenize(word)

            input_ids.extend([self.tokenizer.convert_tokens_to_ids(token) for token in tokenized_word])
            label_ids.append(self.label_map[label])
            label_mask.append(1)

            # the first token gets assigned NER tag and the remaining ones get assigned 'X'
            token_mask_len = len(tokenized_word) - 1
            label_ids.extend([self.label_map['X']] * token_mask_len)
            label_mask.extend([0] * token_mask_len)

        # check the length
        assert len(input_ids) == len(label_ids) == len(label_mask)

        if len(input_ids) >= self.max_len:
            input_ids = input_ids[:(self.max_len - 1)]
            label_ids = label_ids[:(self.max_len - 1)]
            label_mask = label_mask[:(self.max_len - 1)]

        # the first token must be '[SEP]' and the first label must be '[SEP]' too.
        input_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        label_ids.append(self.label_map['[SEP]'])
        label_mask.append(0)

        # check the length again
        assert len(input_ids) == len(label_ids) == len(label_mask)

        sentence_id = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # padding the sequence if its length less than max_len
        padding_len = self.max_len - len(input_ids)
        input_ids.extend([0] * padding_len)
        label_ids.extend([self.label_map['X']] * padding_len)
        label_mask.extend([0] * padding_len)
        sentence_id.extend([0] * padding_len)
        attention_mask.extend([0] * padding_len)

        # since all data are indices, we convert them to torch LongTensors or torch BoolTensor
        input_ids, label_ids, label_mask, attention_mask, sentence_id = torch.LongTensor(input_ids), torch.LongTensor(label_ids),\
            torch.BoolTensor(label_mask), torch.LongTensor(attention_mask), torch.LongTensor(sentence_id)

        # shift tensors to GPU if available
        output = [input_ids, label_ids, attention_mask, sentence_id, label_mask]
        # output = [item.to(self.device) for item in output]

        return output


class UDataset(LDataset):
    pass


class DataLoader:
    """Pytorch DataLoader"""

    def __init__(self, tokenizer, params):
        """
        Args:
            data_dir: (string) directory contains 'train.txt', 'val.txt' and 'test.txt'
            bert_model_dir: (string) for constructing BertTokenizer
            params: Parameters
        """
        self.params = params
        self.tags = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]", "X"]

        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = tokenizer
        self.datasets = self._load_data(os.path.join('data', params.dataset))

    def _load_data(self, data_dir):
        """Load data from 'data_dir'"""

        data = defaultdict(list)

        # read data files
        for data_type in ['train', 'val', 'test']:
            with open(os.path.join(data_dir, data_type + '.txt'), 'r') as f:
                sentence = []
                label = []
                for line in f:
                    # if meets the end of a sentence, add it to data, and clear the cache
                    if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                        if len(sentence) > 0:
                            data[data_type].append((sentence, label))
                            sentence = []
                            label = []
                        continue
                    # if meets a token
                    splits = line.split(' ')
                    sentence.append(splits[0])
                    label.append(splits[-1][:-1])

                if len(sentence) > 0:
                    data[data_type].append((sentence, label))
                    sentence = []
                    label = []

        datasets = {
            'val': LDataset(data['val'], self.tokenizer, self.tag2idx, self.params.max_len),
            'test': LDataset(data['test'], self.tokenizer, self.tag2idx, self.params.max_len)
        }
        train_size = self.params.train_size if self.params.train_size <= 1.0 else 1.0
        train_size = int(train_size * len(data['train']))
        labeled_data = data['train'][:train_size]
        unlabeled_data = data['train'][train_size:]
        datasets['unlabeled'] = UDataset(unlabeled_data, self.tokenizer, self.tag2idx, self.params.max_len)
        if len(labeled_data) > 0:
            datasets['train'] = LDataset(labeled_data, self.tokenizer, self.tag2idx, self.params.max_len)

        return datasets

    def create_iterator(self, data_type, shuffle=True):
        """Create pytorch DataLoaders using datasets"""

        iterator = data.DataLoader(
            dataset=self.dataset[data_type],
            batch_size=self.params.batch_size,
            shuffle=shuffle,
            num_workers=self.params.num_workers
        )

        return iterator
