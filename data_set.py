import os
import torch
from torch.utils import data
from transformers import BertTokenizer


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, segment_ids=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.segment_ids = segment_ids


class Dataset(data.Dataset):
    def __init__(self, data_list, tokenizer, label_map, max_len, device):
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
        label_mask = [0]  # value in (0, 1) - 0 signifies invalid token

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

        input_ids, label_ids, label_mask = input_ids.to(self.device), label_ids.to(self.device), label_mask.to(self.device)
        attention_mask, sentence_id = attention_mask.to(self.device), sentence_id.to(self.device)

        return input_ids, label_ids, attention_mask, sentence_id, label_mask


class DataLoader:
    def __init__(self, data_dir, bert_model_dir, params):
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
        if data_type in ['train', 'val', 'test']:
            examples = self._create_examples(self._readfile(os.path.join(self.data_dir, data_type + ".txt")), data_type)
            return Dataset(examples, self.tokenizer, self.tag2idx, self.max_len, self.device)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")

    def data_iterator(self, dataset, shuffle=False):
        return data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
