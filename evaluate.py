"""Evaluate the model"""

import argparse
import random
import logging
import os

import numpy as np
import torch

import utils
from seqeval.metrics import f1_score, classification_report

from data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='conll',
                    help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=23,
                    help="random seed for initialization")
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--use_crf', action='store_true',
                    help='If stack crf layer on BERT model')


def evaluate(model, data_iterator, params, mark='Eval', verbose=False):
    """Evaluate the model"""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    for batch in data_iterator:
        # fetch the next evaluation batch
        batch = [elem.to(params.device) for elem in batch]
        input_ids, label_ids, attention_mask, sentence_ids, label_mask = batch

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=sentence_ids, attention_mask=attention_mask,
                            label_ids=label_ids, label_masks=label_mask)
            logits, labels = outputs[0], outputs[1]
            batch_output = model.predict(logits, labels)[0]

        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = labels.detach().cpu().numpy()

        batch_true_tags = [
            [idx2tag.get(idx) for idx in indices[np.where(indices != -1)]]
            for indices in batch_tags]
        batch_pred_tags = [
            [idx2tag.get(idx)
             for idx in indices[np.where(batch_tags[i] != -1)]]
            for i, indices in enumerate(batch_output)]

        true_tags.extend(batch_true_tags)
        pred_tags.extend(batch_pred_tags)

    assert len(pred_tags) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v)
                            for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics   


def main():
    args = parser.parse_args()

    model_dir = 'experiments/' + args.dataset
    # Load the parameters from json file
    json_path = os.path.join(model_dir, 'params.json')
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    data_loader = DataLoader(os.path.join(
        'data', params.dataset), params.bert_model_dir, params)

    # Prepare model
    if params.use_crf is True:
        logging.info('Use CRF layer on BERT')
        from model import BertCRFForSequenceTagging as BertForSequenceTagging
    else:
        from model import BertForSequenceTagging
    model = BertForSequenceTagging.from_pretrained(
        model_dir, num_labels=len(params.tag2idx))
    model.to(params.device)

    if args.fp16:
        model.half()
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    logging.info("\n>>> Starting evaluation...")
    test_iter = data_loader.data_iterator('test', shuffle=False)
    evaluate(model, test_iter, params, mark='Test', verbose=True)


if __name__ == '__main__':
    main()
