from transformers import BertPreTrainedModel, BertModel, BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from torchcrf import CRF


class BertOnlyForSequenceTagging(BertForTokenClassification):
    """Only use Bert for sequence tagging, without other layers"""

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, label_masks=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        # obtain original token representations from sub_words representations (by selecting the first sub_word)
        origin_sequence_output = [
            layer[mask]
            for layer, mask in zip(sequence_output, label_masks)]

        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True, padding_value=-1)

        padded_sequence_output = self.dropout(padded_sequence_output)
        logits = self.classifier(padded_sequence_output)

        outputs = (logits,)
        if labels is not None:
            labels = [label[mask] for mask, label in zip(label_masks, labels)]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
            mask = (labels != -1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss /= mask.float().sum()
            outputs = (loss,) + outputs + (labels,)

        return outputs  # (loss), scores


class BertCRFForSequenceTagging(BertPreTrainedModel):
    """Use Bert and CRF for sequence tagging"""

    def __init__(self, config):
        super(BertCRFForSequenceTagging, self).__init__(config)
        self.num_labels = config.num_labels
        self.device = 'cpu'
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(config.num_labels, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        """Use crf for sequence tagging.

        For example, in the case of max_seq_length=10:
          raw_data:          你 是 一 个 人 le
          token:       [CLS] 你 是 一 个 人 ##le [SEP]
          input_ids:     101 2  12 13 16 14 15   102   0 0
          attention_mask:  1 1  1  1  1  1   1     1   0 0
          labels:            T  T  O  O  O
          starts:          0 1  1  1  1  1   0     0   0 0

        starts means 'label_masks', it can be used for mask in crf.
        """
        input_ids, label_masks = input_data

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        # obtain original token representations from sub_words representations (by selecting the first sub_word)
        origin_sequence_output = []
        origin_sequence_mask = []
        for layer, starts in zip(sequence_output, label_masks):
            one_sequence_out = layer[starts.nonzero().squeeze(1)]
            one_sequence_mask = torch.ones(one_sequence_out.size(
                0), dtype=torch.uint8).to(self.device)
            origin_sequence_output.append(one_sequence_out)
            origin_sequence_mask.append(one_sequence_mask)

        padded_sequence_output = pad_sequence(
            origin_sequence_output, batch_first=True)
        padded_sequence_mask = pad_sequence(
            origin_sequence_mask, batch_first=True)

        padded_sequence_output = self.dropout(padded_sequence_output)
        emissions = self.classifier(padded_sequence_output)

        outputs = (emissions,)
        if labels is not None:  # For training
            loss = -1 * self.crf(emissions, labels, mask=padded_sequence_mask)
            outputs = (loss,) + outputs
        else:  # For evaluation
            best_tags = self.crf.decode(emissions, padded_sequence_mask)
            outputs = (best_tags,)

        return outputs  # (loss), scores
