import os
import torch
from torch.utils import data
from transformers import BertTokenizer


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, segment_ids=None):
        """Constructs a InputExample.
        Args:
          guid: (string) Unique id for the example.
          text: (string) The untokenized text of the sequence.
          label: (string) The label of the example.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.segment_ids = segment_ids


class Dataset(data.Dataset):
    """A torch dataset for iterating"""

    def __init__(self, data_list, tokenizer, label_map, max_len, device):
        """ Construct a dataset for training/evaluating.

        Args:
            data_list: (list) list of InputExample
            tokenizer: (BertTokenizer) tokenize words
            label_map: (dict) convert tags to ids
            max_len: (int) maximum length of sequences
            device: (string) cpu or gpu
        """
        self.max_len = max_len
        self.label_map = label_map
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_example = self.data_list[idx]
        text = input_example.text
        label = input_example.label

        word_tokens = ['[CLS]']
        label_list = ['[CLS]']
        label_mask = [0]
        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')]
        label_ids = [self.label_map['[CLS]']]

        # iterate over individual tokens and their labels
        for word, label in zip(text.split(), label):
            tokenized_word = self.tokenizer.tokenize(word)

            for token in tokenized_word:
                word_tokens.append(token)
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))

            label_list.append(label)
            label_ids.append(self.label_map[label])
            label_mask.append(1)

            # the first token gets assigned NER tag and the remaining ones get assigned 'X'
            for i in range(1, len(tokenized_word)):
                label_list.append('X')
                label_ids.append(self.label_map['X'])
                label_mask.append(0)

        # check the length
        assert len(word_tokens) == len(label_list)
        assert len(word_tokens) == len(input_ids)
        assert len(word_tokens) == len(label_ids)
        assert len(word_tokens) == len(label_mask)

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

        # check the length again
        assert len(word_tokens) == len(label_list)
        assert len(word_tokens) == len(input_ids)
        assert len(word_tokens) == len(label_ids)
        assert len(word_tokens) == len(label_mask)

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

        # since all data are indices, we convert them to torch LongTensors or torch BoolTensor
        input_ids, label_ids, label_mask, attention_mask, sentence_id = torch.LongTensor(input_ids), torch.LongTensor(label_ids),\
            torch.BoolTensor(label_mask), torch.LongTensor(attention_mask), torch.LongTensor(sentence_id)

        # shift tensors to GPU if available
        output = [input_ids, label_ids, attention_mask, sentence_id, label_mask]
        output = [item.to(self.device) for item in output]

        return output


class DataLoader:
    """Pytorch DataLoader"""

    def __init__(self, data_dir, bert_model_dir, params):
        """
        Args:
            data_dir: (string) directory contains 'train.txt', 'val.txt' and 'test.txt'
            bert_model_dir: (string) for constructing BertTokenizer
            params: Parameters
        """
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.workers_num = params.num_workers
        self.tags = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]", "X"]

        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=False)

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            label = label
            examples.append(InputExample(guid=guid, text=text_a, label=label))
        return examples

    def _readfile(self, filename):
        """Read a dataset file

        Args:
            filename: (string) train.txt, val.txt or test.txt

        Return:
            data: (list) (sentence, label) tuples
        """
        f = open(filename)
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
        return data

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (Dataset) an instance of pytorch Dataset class
        """
        if data_type in ['train', 'val', 'test']:
            examples = self._create_examples(self._readfile(os.path.join(self.data_dir, data_type + ".txt")), data_type)
            return Dataset(examples, self.tokenizer, self.tag2idx, self.max_len, self.device)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")

    def data_iterator(self, dataset, shuffle=False):
        """Create a pytorch DataLoader given a dataset

        Args:
            dataset: (data.Dataset) a pytorch Dataset class instance
            shuffle: (bool) whether shuffle the dataset

        Return:
            an instance of pytorch DataLoader class
        """
        return data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
