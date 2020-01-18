"""Train and evaluate the model"""

import argparse
import random
import logging
import os
import numpy as np
import torch.nn.functional as F

import torch
import transformers
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdamW
from tqdm import trange
from seqeval.metrics import f1_score, classification_report

from sequence_tagger import BertOnlyForSequenceTagging as BertForSequenceTagging
from data_loader import DataLoader
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='toy', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/conll/'")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                    "0 (default value): dynamic loss scaling.\n"
                    "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--full_finetuning', action='store_true', help='BERT: If full finetuning bert model')
parser.add_argument('--max_len', default=128, type=int, help='BERT: maximul sequence lenghth')
parser.add_argument('--learning_rate', default=5e-5, type=float, help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay for the optimizer')
parser.add_argument('--clip_grad', default=1.0, type=float, help='Gradient clipping')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Warmup configuration for the optimizer')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Configuration for the optimizer')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training and testing')
parser.add_argument('--max_epoch', default=10, type=int, help='Number of maximum epochs')
parser.add_argument('--min_epoch', default=5, type=int, help='Number of minimum epochs')
parser.add_argument('--patience', default=0.02, type=float, help='Increasement between two epochs')
parser.add_argument('--patience_num', default=5, type=int, help='Early stopping creteria')


def train(model, data_iterator, optimizer, scheduler, params):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()
    # scheduler.step()

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    one_epoch = trange(params.train_steps)
    for step, batch in zip(one_epoch, data_iterator):
        # fetch the next training batch
        input_ids, label_ids, attention_mask, sentence_ids, label_mask = batch

        # compute model output and loss
        loss = model(input_ids, token_type_ids=sentence_ids,
                     attention_mask=attention_mask, labels=label_ids, label_masks=label_mask)[0]

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
        scheduler.step()
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()

        # update the average loss
        loss_avg.update(loss.item())
        one_epoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def evaluate(model, data_iterator, params, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    one_epoch = trange(params.eval_steps)
    for step, batch in zip(one_epoch, data_iterator):
        # fetch the next evaluation batch
        input_ids, label_ids, attention_mask, sentence_ids, label_mask = batch

        with torch.no_grad():
            loss, logits, labels = model(input_ids, token_type_ids=sentence_ids,
                                         attention_mask=attention_mask, labels=label_ids, label_masks=label_mask)
        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()
        loss_avg.update(loss.item())

        batch_output = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = labels.to('cpu').numpy()

        batch_true_tags = [
            [idx2tag.get(idx) for idx in indices[np.where(indices != -1)]]
            for indices in batch_tags]
        batch_pred_tags = [
            [idx2tag.get(idx) for idx in indices[np.where(batch_tags[i] != -1)]]
            for i, indices in enumerate(batch_output)]

        true_tags.extend(batch_true_tags)
        pred_tags.extend(batch_pred_tags)

        one_epoch.set_postfix(eval_loss='{:05.3f}'.format(loss_avg()))

    assert len(pred_tags) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v)
                            for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics


def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, model_dir, restore_dir=None):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_dir if specified
    if restore_dir is not None:
        model = BertForSequenceTagging.from_pretrained(restore_dir)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        params.train_steps = params.train_size // params.batch_size
        params.val_steps = params.val_size // params.batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(
            train_data, shuffle=True)

        # Train for one epoch on training set
        train(model, train_data_iterator, optimizer, scheduler, params)

        # data iterator for evaluation
        # train_data_iterator = data_loader.data_iterator(train_data, shuffle=False)
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        # params.eval_steps = params.train_steps
        # train_metrics = evaluate(model, train_data_iterator, params, mark='Train')
        params.eval_steps = params.val_steps
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
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
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

    # Initialize the DataLoader
    data_dir = 'data/' + args.dataset
    if args.dataset in ["conll"]:
        bert_model_dir = 'pretrained_bert_models/bert-base-cased/'
    elif args.dataset in ["msra"]:
        bert_model_dir = 'pretrained_bert_models/bert-base-chinese/'

    data_loader = DataLoader(data_dir, bert_model_dir, params)

    # Load training data and test data
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('val')

    # Specify the training and validation dataset sizes
    params.train_size = train_data.__len__()
    params.val_size = val_data.__len__()

    # Prepare model
    model = BertForSequenceTagging.from_pretrained(bert_model_dir, num_labels=len(params.tag2idx))
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
        # optimizer = AdamW(model.parameters(), lr=params.learning_rate, correct_bias=False)
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=params.learning_rate)
        train_steps_per_epoch = params.train_size // params.batch_size
        warmup_steps = int(params.warmup_proportion * params.epoch_num * train_steps_per_epoch)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=params.epoch_num * train_steps_per_epoch)

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, train_data, val_data, optimizer,
                       scheduler, params, model_dir, args.restore_dir)
