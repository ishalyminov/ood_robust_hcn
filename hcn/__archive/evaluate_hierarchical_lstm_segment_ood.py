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
from modules.lstm_net import HierarchicalLSTM
from modules.actions import ActionTracker
from modules.util import read_dialogs, get_utterances, make_dataset_for_hierarchical_lstm, make_vocabulary, BoW_encoder, load_model
from utils.training_utils import batch_generator
from babi_tools.ood_augmentation import DEFAULT_CONFIG_FILE

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)

with open(DEFAULT_CONFIG_FILE) as babi_config_in:
    BABI_CONFIG = json.load(babi_config_in)


def evaluate(in_model, in_dataset):
    X, context_features, action_masks, y = in_dataset
    losses, predictions = [], []
    batch_gen = batch_generator([X, context_features, action_masks, y], 64, verbose=False)
    for batch_x, batch_action_masks, batch_sequence_masks, batch_y in batch_gen:
        batch_predictions, batch_losses = in_model.forward(batch_x, batch_action_masks, batch_sequence_masks, batch_y)
        losses += batch_losses.tolist()
        predictions += batch_predictions.flatten().tolist()
    return (sum([gold_i == pred_i and gold_i != PAD_ID for gold_i, pred_i in zip(y.flatten(), predictions)]) / np.sum(list(map(lambda x: int(x != 0), y.flatten()))),
            np.mean(losses))


def evaluate_advanced(in_model, in_dataset, in_action_templates, post_ood_turns=None, ignore_ood_accuracy=False):
    if BABI_CONFIG['backoff_utterance'].lower() in in_action_templates:
        backoff_action = in_action_templates.index(BABI_CONFIG['backoff_utterance'].lower())
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
    total_post_ood_turns = 0
    correct_post_ood_turns = 0
    correct_turns_after_ood = 0
    for y_true, y_pred in zip(y_true_dialog, y_pred_dialog):
        ood_occurred = False
        prev_action = None
        for y_i_true, y_i_pred in zip(y_true, y_pred):
            current_turn_counts = not (y_i_true == backoff_action and ignore_ood_accuracy)
            total_turns += int(current_turn_counts)
            correct_turns += int(y_i_true == y_i_pred and current_turn_counts)
            if prev_action == backoff_action and y_i_true != backoff_action:
                total_post_ood_turns += 1
                correct_post_ood_turns += int(y_i_true ==y_i_pred)
            if ood_occurred:
                total_turns_after_ood += int(current_turn_counts)
                correct_turns_after_ood += int(y_i_true == y_i_pred and current_turn_counts)
            if y_i_true == backoff_action:
                ood_occurred = True
            prev_action = y_i_true
    post_ood_correct, post_ood_total = 0, 0
    if post_ood_turns is not None:
        non_pad_counter = 0
        post_ood_true, post_ood_pred = [], []
        for y_true_i, y_pred_i in zip(np.array(y).flatten(), np.array(predictions).flatten()):
            if y_true_i == PAD_ID:
                continue
            if non_pad_counter in post_ood_turns:
                post_ood_true.append(y_true_i)
                post_ood_pred.append(y_pred_i)
            non_pad_counter += 1
        post_ood_correct = sum([int(true_i == pred_i) for true_i, pred_i in zip(post_ood_true, post_ood_pred)])
        post_ood_total = len(post_ood_turns)

    return {'avg_loss': np.mean(losses),
            'correct_turns': correct_turns,
            'total_turns': total_turns,
            'correct_turns_after_ood': correct_turns_after_ood,
            'total_turns_after_ood': total_turns_after_ood,
            'total_post_ood_turns': post_ood_total,
            'correct_post_ood_turns': post_ood_correct}


def mark_post_ood_turns(in_turns):
    total_turn_counter, ind_turn_counter = 0, 0
    prev_answer = None
    backoff_utterance = BABI_CONFIG['backoff_utterance'].lower()
    post_ood_turn_indices_clean, post_ood_turn_indices_noisy = set([]), set([])
    for user_turn, system_turn in in_turns:
        if system_turn != backoff_utterance:
            if prev_answer == backoff_utterance:
                post_ood_turn_indices_clean.add(ind_turn_counter)
                post_ood_turn_indices_noisy.add(total_turn_counter)
            ind_turn_counter += 1
        total_turn_counter += 1
        prev_answer = system_turn
    return post_ood_turn_indices_clean, post_ood_turn_indices_noisy


def main(in_clean_dataset_folder, in_noisy_dataset_folder, in_model_folder, in_no_ood_evaluation):
    rev_vocab, kb, action_templates, config = load_model(in_model_folder)
    clean_dialogs, clean_indices = read_dialogs(os.path.join(in_clean_dataset_folder, 'dialog-babi-task6-dstc2-tst.txt'),
                                                with_indices=True)
    noisy_dialogs, noisy_indices = read_dialogs(os.path.join(in_noisy_dataset_folder, 'dialog-babi-task6-dstc2-tst.txt'),
                                                with_indices=True)

    post_ood_turns_clean, post_ood_turns_noisy = mark_post_ood_turns(noisy_dialogs)

    assert len(post_ood_turns_clean) == len(post_ood_turns_noisy)

    for post_ood_turn_clean, post_ood_turn_noisy in zip(sorted(post_ood_turns_clean), sorted(post_ood_turns_noisy)):
        noisy_dialogs[post_ood_turn_noisy][0] = clean_dialogs[post_ood_turn_clean][0]
    et = EntityTracker(kb)
    at = ActionTracker(None, et)
    at.set_action_templates(action_templates)

    vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    X_clean, context_features_clean, action_masks_clean, y_clean = make_dataset_for_hierarchical_lstm(clean_dialogs,
                                                                                                      clean_indices,
                                                                                                      vocab,
                                                                                                      et,
                                                                                                      at,
                                                                                                      **config)
    X_noisy, context_features_noisy, action_masks_noisy, y_noisy = make_dataset_for_hierarchical_lstm(noisy_dialogs,
                                                                                                      noisy_indices,
                                                                                                      vocab,
                                                                                                      et,
                                                                                                      at,
                                                                                                      **config)

    net = HierarchicalLSTM(config, context_features_clean.shape[-1], action_masks_clean.shape[-1])
    net.restore(in_model_folder)
    eval_stats_clean = evaluate_advanced(net,
                                         (X_clean, context_features_clean, action_masks_clean, y_clean),
                                         at.action_templates,
                                         post_ood_turns=post_ood_turns_clean)
    print('Clean dataset: {} turns overall'.format(eval_stats_clean['total_turns']))
    print('Accuracy:')
    accuracy = eval_stats_clean['correct_turns'] / eval_stats_clean['total_turns']
    accuracy_post_ood = eval_stats_clean['correct_post_ood_turns'] / eval_stats_clean['total_post_ood_turns'] \
        if eval_stats_clean['total_post_ood_turns'] != 0 \
        else 0
    print('overall: {:.3f}; directly post-OOD: {:.3f}'.format(accuracy, accuracy_post_ood))
    print('Loss : {:.3f}'.format(eval_stats_clean['avg_loss']))

    eval_stats_noisy = evaluate_advanced(net,
                                         (X_noisy, context_features_noisy, action_masks_noisy, y_noisy),
                                         at.action_templates,
                                         post_ood_turns=post_ood_turns_noisy)
    print('\n\n')
    print('Noisy dataset: {} turns overall, {} turns after the first OOD'.format(eval_stats_noisy['total_turns'],
                                                                                 eval_stats_noisy['total_turns_after_ood']))
    print('Accuracy:')
    accuracy = eval_stats_noisy['correct_turns'] / eval_stats_noisy['total_turns']
    accuracy_after_ood = eval_stats_noisy['correct_turns_after_ood'] / eval_stats_noisy['total_turns_after_ood'] \
        if eval_stats_noisy['total_turns_after_ood'] != 0 \
        else 0
    accuracy_post_ood = eval_stats_noisy['correct_post_ood_turns'] / eval_stats_noisy['total_post_ood_turns'] \
        if eval_stats_noisy['total_post_ood_turns'] != 0 \
        else 0
    print('overall: {:.3f}; after first OOD: {:.3f}, directly post-OOD: {:.3f}'.format(accuracy,
                                                                                       accuracy_after_ood,
                                                                                       accuracy_post_ood))
    print('Loss : {:.3f}'.format(eval_stats_noisy['avg_loss']))

    if in_no_ood_evaluation:
        eval_stats_no_ood = evaluate_advanced(net,
                                              (X_noisy, context_features_noisy, action_masks_noisy, y_noisy),
                                              at.action_templates,
                                              post_ood_turns=post_ood_turns_noisy,
                                              ignore_ood_accuracy=True)
        print('Accuracy (OOD turns ignored):')
        accuracy = eval_stats_no_ood['correct_turns'] / eval_stats_no_ood['total_turns']
        accuracy_after_ood = eval_stats_no_ood['correct_turns_after_ood'] / eval_stats_no_ood['total_turns_after_ood'] \
            if eval_stats_no_ood['total_turns_after_ood'] != 0 \
            else 0
        accuracy_post_ood = eval_stats_no_ood['correct_post_ood_turns'] / eval_stats_no_ood['total_post_ood_turns'] \
            if eval_stats_no_ood['total_post_ood_turns'] != 0 \
            else 0
        print('overall: {:.3f}; after first OOD: {:.3f}, directly post-OOD: {:.3f}'.format(accuracy, accuracy_after_ood, accuracy_post_ood))


def configure_argument_parser():
    result_parser = ArgumentParser('Evaluate a trained Hybrid Code Network on bAbI Dialog Tasks data')
    result_parser.add_argument('clean_dataset_folder')
    result_parser.add_argument('noisy_dataset_folder')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('--perform_no_ood_eval', default=False, action='store_true')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.clean_dataset_folder, args.noisy_dataset_folder, args.model_folder, args.perform_no_ood_eval)

