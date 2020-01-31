import torch
import torch.nn.functional as F

from transformers import BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torchcrf import CRF


class BertForSequenceTagging(BertForTokenClassification):
    """Only use Bert for sequence tagging, without other layers"""

    def __init__(self, config):
        super(BertForSequenceTagging, self).__init__(config)

        self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                head_mask=None, label_ids=None, label_masks=None, output_hidden=False):

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

        # obtain original labels and pad them
        labels = [label[mask] for mask, label in zip(label_masks, label_ids)]
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

        out = (logits, padded_labels) if output_hidden is False else (logits, padded_labels, outputs[1])
        return out

    def loss(self, logits, labels):

        mask = (labels != -1)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss /= mask.float().sum()

        return loss

    def predict(self, logits, labels):

        max_probs, output = torch.max(F.softmax(logits, dim=2), dim=2)

        mask = (labels != -1).float()
        reverse_mask = (labels == -1).float()
        confidence = torch.prod(max_probs * mask + reverse_mask, dim=1)

        return (output, confidence)


class BertCRFForSequenceTagging(BertForSequenceTagging):

    def __init__(self, config):
        super(BertCRFForSequenceTagging, self).__init__(config)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def loss(self, logits, labels):

        mask = (labels != -1)
        loss = -1 * self.crf(F.log_softmax(logits, dim=2), labels, mask=mask)
        loss /= mask.float().sum()

        return loss

    def predict(self, logits, labels):

        mask = (labels != -1)
        emissions = F.log_softmax(logits, dim=2)
        output = self.crf.decode(emissions, mask=mask)
        output = [torch.tensor(x, dtype=labels.dtype, device=labels.device) for x in output]
        padded_output = pad_sequence(output, batch_first=True, padding_value=-1)
        confidence = self.crf(emissions, padded_output, mask=mask, reduction='none')

        return (padded_output, confidence)
