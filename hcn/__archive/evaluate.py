from __future__ import print_function

import os
from argparse import ArgumentParser
import random
import sys
import json

import numpy as np
import tensorflow as tf

from modules.util import PAD_ID, UNK_ID

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.entities import EntityTracker
from modules.lstm_net import LSTM_net
from modules.actions import ActionTracker
from modules.util import read_dialogs, get_utterances, make_dataset, make_vocabulary, BoW_encoder, load_model
from utils.training_utils import batch_generator
from babi_tools.ood_augmentation import DEFAULT_CONFIG_FILE 

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)

with open(DEFAULT_CONFIG_FILE) as babi_config_in:
    BABI_CONFIG = json.load(babi_config_in)


def evaluate(in_model, in_dataset):
    X, action_masks, sequence_masks, y = in_dataset
    losses, predictions = [], []
    batch_gen = batch_generator([X, action_masks, sequence_masks, y], 64, verbose=False)
    for batch_x, batch_action_masks, batch_sequence_masks, batch_y in batch_gen:
        batch_predictions, batch_losses = in_model.forward(batch_x, batch_action_masks, batch_sequence_masks, batch_y)
        losses += batch_losses.tolist()
        predictions += batch_predictions.flatten().tolist()
    return (sum([gold_i == pred_i and gold_i != PAD_ID for gold_i, pred_i in zip(y.flatten(), predictions)]) / np.sum(map(lambda x: int(x != 0), y.flatten())),
            np.mean(losses))


def evaluate_advanced(in_model, in_dataset, in_dialog_indices, in_action_templates, ignore_ood_accuracy=False):
    if BABI_CONFIG['backoff_utterance'].lower() in in_action_templates:
        backoff_action = in_action_templates.index(babi_config['backoff_utterance'].lower())
    else:
        backoff_action = UNK_ID

    X, action_masks, sequence_masks, y = in_dataset
    losses, predictions = [], []
    batch_gen = batch_generator([X, action_masks, sequence_masks, y], 64, verbose=False)
    for batch_x, batch_action_masks, batch_sequence_masks, batch_y in batch_gen:
        batch_predictions, batch_losses = in_model.forward(batch_x, batch_action_masks, batch_sequence_masks, batch_y)
        losses += list(batch_losses)
        predictions += batch_predictions.tolist()

    y_true_dialog, y_pred_dialog = [], []
    for y_true_i, y_pred_i in zip(y, predictions):
        y_true_dialog_i, y_pred_dialog_i = [], []
        for y_true_i_j, y_pred_i_j in zip(y_true_i, y_pred_i):
            if y_true_i_j != PAD_ID:
                y_true_dialog_i.append(y_true_i_j)
                y_pred_dialog_i.append(y_pred_i_j)
        y_true_dialog.append(y_true_dialog_i)
        y_pred_dialog.append(y_pred_dialog_i)

    total_turns = 0
    correct_turns = 0
    total_turns_after_ood = 0
    correct_turns_after_ood = 0
    for y_true, y_pred in zip(y_true_dialog, y_pred_dialog):
        ood_occurred = False
        for y_i_true, y_i_pred in zip(y_true, y_pred):
            current_turn_counts = not (y_i_true == backoff_action and ignore_ood_accuracy)
            total_turns += int(current_turn_counts)
            correct_turns += int(y_i_true == y_i_pred and current_turn_counts)
            if ood_occurred:
                total_turns_after_ood += int(current_turn_counts)
                correct_turns_after_ood += int(y_i_true == y_i_pred and current_turn_counts)
            if y_i_true == backoff_action:
                ood_occurred = True
    return {'avg_loss': np.mean(losses),
            'correct_turns': correct_turns,
            'total_turns': total_turns,
            'correct_turns_after_ood': correct_turns_after_ood,
            'total_turns_after_ood': total_turns_after_ood}


def main(in_dataset_folder, in_model_folder, in_no_ood_evaluation):
    rev_vocab, kb, action_templates, config = load_model(in_model_folder)
    test_dialogs, test_indices = read_dialogs(os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-tst.txt'),
                                              with_indices=True)

    et = EntityTracker(kb)
    at = ActionTracker(None, et)
    at.set_action_templates(action_templates)

    vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    X, action_masks, sequence_masks, y = make_dataset(test_dialogs,
                                                      test_indices,
                                                      vocab,
                                                      et,
                                                      at,
                                                      config['max_input_length'])

    net = LSTM_net(config, X.shape[-1], action_masks.shape[-1])
    net.restore(in_model_folder)
    eval_stats_full_dataset = evaluate_advanced(net,
                                                (X, action_masks, sequence_masks, y),
                                                test_indices,
                                                at.action_templates)
    print('Full dataset: {} turns overall, {} turns after the first OOD'.format(eval_stats_full_dataset['total_turns'],
                                                                                 eval_stats_full_dataset['total_turns_after_ood']))
    print('Accuracy:')
    accuracy = eval_stats_full_dataset['correct_turns'] / eval_stats_full_dataset['total_turns']
    accuracy_after_ood = eval_stats_full_dataset['correct_turns_after_ood'] / eval_stats_full_dataset['total_turns_after_ood'] \
        if eval_stats_full_dataset['total_turns_after_ood'] != 0 \
        else 0
    print('overall: {:.3f}; after first OOD: {:.3f}'.format(accuracy, accuracy_after_ood))
    print('Loss : {:.3f}'.format(eval_stats_full_dataset['avg_loss']))

    if in_no_ood_evaluation:
        eval_stats_no_ood = evaluate_advanced(net,
                                              (X, action_masks, sequence_masks, y),
                                              test_indices,
                                              at.action_templates,
                                              ignore_ood_accuracy=True)
        print('Accuracy (OOD turns ignored):')
        accuracy = eval_stats_no_ood['correct_turns'] / eval_stats_no_ood['total_turns']
        accuracy_after_ood = eval_stats_no_ood['correct_turns_after_ood'] / eval_stats_no_ood['total_turns_after_ood'] \
            if eval_stats_no_ood['total_turns_after_ood'] != 0 \
            else 0
        print('overall: {:.3f}; after first OOD: {:.3f}'.format(accuracy, accuracy_after_ood))
        print('Loss : {:.3f}'.format(eval_stats_no_ood['avg_loss']))


def configure_argument_parser():
    result_parser = ArgumentParser('Evaluate a trained Hybrid Code Network on bAbI Dialog Tasks data')
    result_parser.add_argument('dataset_folder')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('--perform_no_ood_eval', default=False, action='store_true')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.model_folder, args.perform_no_ood_eval)

