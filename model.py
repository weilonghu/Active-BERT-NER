from transformers import BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss


class BertOnlyForSequenceTagging(BertForTokenClassification):
    """Only use Bert for sequence tagging, without other layers"""

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
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

        return logits

    def loss(self, logits, labels, label_masks):

        labels = [label[mask] for mask, label in zip(label_masks, labels)]
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
        mask = (labels != -1)
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        loss /= mask.float().sum()
        outputs = (loss, labels,)

        return outputs
