"""Train and evaluate the model"""

import argparse
import random
import logging
import os
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdamW
from tqdm import tqdm, trange
from seqeval.metrics import f1_score, classification_report

from model import BertOnlyForSequenceTagging as BertForSequenceTagging
from data_loader import DataLoader
from strategy.strategy import Strategy
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='conll', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/conll/'")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                    "0 (default value): dynamic loss scaling.\n"
                    "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--full_finetuning', action='store_false', help='BERT: If full finetuning bert model')
parser.add_argument('--max_len', default=128, type=int, help='BERT: maximul sequence lenghth')
parser.add_argument('--bert_model_dir', default='pretrained_bert_models', type=str, help='BERT: directory containing BERT model')
parser.add_argument('--learning_rate', default=5e-5, type=float, help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay for the optimizer')
parser.add_argument('--clip_grad', default=1.0, type=float, help='Gradient clipping')
parser.add_argument('--warmup_steps', default=250, type=int, help='Warmup configuration for the optimizer')
parser.add_argument('--min_lr_ratio', default=0.5, type=float, help='Minimum learning rate')
parser.add_argument('--decay_steps', default=1500, type=int, help="Decay steps for LambdLR scheduler")
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Configuration for the optimizer')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training and testing')
parser.add_argument('--num_query', default=100, type=int, help='Number of queried batches with active learning')
parser.add_argument('--num_epoch', default=1, type=int, help='Number of training epochs')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for pytorch DataLoader')
parser.add_argument('--train_size', default=1, type=float, help='Proportion of train dataset for initialized training')
parser.add_argument('--do_train', action='store_false', help='If train the model')
parser.add_argument('--do_test', action='store_false', help='If test the model')
parser.add_argument('--eval_every', default=5, type=int, help='Evaluate the model every "eval_every" batchs')
parser.add_argument('--incremental_train', action='store_true', help='When selecting a batch actively, if only train on the batch')
parser.add_argument('--active_strategy', default='random_select', type=str, help='Strategy name used in active learning')


def train(model, data_iterator, optimizer, params):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()

    tqdm_bar = trange(params.num_epoch)
    for _ in tqdm_bar:
        # a running average object for loss
        loss_avg = utils.RunningAverage()

        for batch in data_iterator:
            # fetch the next training batch
            batch = [elem.to(params.device) for elem in batch]
            input_ids, label_ids, attention_mask, sentence_ids, label_mask = batch

            # compute model output and loss
            logits = model(input_ids, token_type_ids=sentence_ids,
                           attention_mask=attention_mask, label_masks=label_mask)
            loss, _ = model.loss(logits=logits, labels=label_ids, label_masks=label_mask)

            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=params.clip_grad)

            # performs updates using calculated gradients
            optimizer.step()
            # clear previous gradients, compute gradients of all variables wrt loss
            model.zero_grad()

            # update the average loss
            loss_avg.update(loss.item())
            tqdm_bar.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    return loss_avg()


def evaluate(model, data_iterator, params, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    for batch in tqdm(data_iterator):
        # fetch the next evaluation batch
        batch = [elem.to(params.device) for elem in batch]
        input_ids, label_ids, attention_mask, sentence_ids, label_mask = batch

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=sentence_ids,
                           attention_mask=attention_mask, label_masks=label_mask)
            loss, labels = model.loss(logits=logits, labels=label_ids, label_masks=label_mask)
        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()

        batch_output = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = labels.detach().cpu().numpy()

        batch_true_tags = [
            [idx2tag.get(idx) for idx in indices[np.where(indices != -1)]]
            for indices in batch_tags]
        batch_pred_tags = [
            [idx2tag.get(idx) for idx in indices[np.where(batch_tags[i] != -1)]]
            for i, indices in enumerate(batch_output)]

        true_tags.extend(batch_true_tags)
        pred_tags.extend(batch_pred_tags)

    assert len(pred_tags) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v)
                            for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics


def train_active(model, data_loader, optimizer, params, model_dir):
    """Train the model and evaluate every epoch."""

    best_val_f1 = 0.0
    patience_counter = 0
    val_data_iterator = data_loader.data_iterator('val', shuffle=False)
    strategy_func = Strategy().get_strategy_proxy(params.active_strategy)

    for query in range(1, params.num_query + 1):

        num_unlabeled_data = data_loader.unlabled_length()
        if num_unlabeled_data > 0:
            # Predict probs of unlabeled data
            unlabeled_logits = []
            unlabeled_iter = data_loader.data_iterator('unlabeled', shuffle=False)
            for batch in unlabeled_iter:
                batch = [elem.to(params.device) for elem in batch]
                input_ids, label_ids, attention_mask, sentence_ids, label_mask = batch
                logits = model(input_ids, token_type_ids=sentence_ids,
                               attention_mask=attention_mask, label_masks=label_mask)
                unlabeled_logits.append(logits)
            unlabeled_logits = torch.cat(unlabeled_logits, dim=0)
            del unlabeled_iter

            # Query a batch of unlabeled data, then put into train set
            query_num = min(num_unlabeled_data, params.batch_size)
            query_idx = strategy_func(unlabeled_logits, None, query_num)
            data_loader.update_dataset(query_idx)

        # Train on new training data
        if params.incremental_train and num_unlabeled_data > 0:
            query_data = []
            for idx in query_idx:
                query_data.append(data_loader.datasets['unlabeled'].get(idx))
            train_iter = iter(query_data)
        else:
            train_iter = data_loader.data_iterator('train', shuffle=True)
        train(model, train_iter, optimizer, params)

        # Evaluate for val dataset, perform early stopping
        if query % params.eval_every:
            val_metrics = evaluate(model, val_data_iterator, params, mark='Val')

            val_f1 = val_metrics['f1']
            improve_f1 = val_f1 - best_val_f1
            if improve_f1 > 0:
                logging.info("- Found new best F1")
                best_val_f1 = val_f1
                model.save_pretrained(model_dir)
                if improve_f1 < params.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping and logging best f1
            if (patience_counter >= params.patience_num and query > params.min_epoch_num) or query == params.epoch_num:
                logging.info("Early stop, Best val f1: {:05.2f}".format(best_val_f1))
                break


if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = 'experiments/' + args.dataset
    # save the parameters to json file
    json_path = os.path.join(model_dir, 'params.json')
    utils.save_json(vars(args), json_path)
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
    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logging.info("device: {}, n_gpu: {}, 16-bits training: {}".format(params.device, params.n_gpu, args.fp16))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    data_loader = DataLoader(os.path.join('data', params.dataset), params.bert_model_dir, params)

    # Prepare model
    model = BertForSequenceTagging.from_pretrained(params.bert_model_dir, num_labels=len(params.tag2idx))
    model.to(params.device)
    if args.fp16:
        model.half()

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': params.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else:  # only finetune the head classifier
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer]}]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "lease install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=params.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=params.learning_rate)

    # restore the model
    if params.restore_dir is not None:
        model = BertForSequenceTagging.from_pretrained(params.restore_dir)
        logging.info('Restore model from {}'.format(params.restore_dir))

    # Train and evaluate the model
    if params.do_train:
        if params.train_size > 0:
            logging.info("Starting training for {} epoch(s)".format(params.num_epoch))
            train_iter = data_loader.data_iterator('train', shuffle=True)
            train(model, train_iter, optimizer, params)
            del train_iter

        if params.train_size < 1:
            logging.info('Start training using active learning...')
            train_active(model, data_loader, optimizer, params, model_dir)

    if params.do_test:
        logging.info('Starting testing the model...')
        test_iter = data_loader.data_iterator('test', shuffle=False)
        evaluate(model, test_iter, params, mark='Test', verbose=True)
