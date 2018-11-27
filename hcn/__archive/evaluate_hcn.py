from __future__ import print_function

import os
from argparse import ArgumentParser
import random
import sys
import json

import numpy as np

random.seed(273)
np.random.seed(273)

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.entities import EntityTracker
from modules.dialog_level_lstm_v2 import DialogLevelLSTMv2
from modules.actions import ActionTracker
from modules.util import read_dialogs, get_utterances, load_model
from modules.evaluation import evaluate, evaluate_advanced, mark_post_ood_turns
from utils.preprocessing import make_dataset_for_vhcn_v2, make_vocabulary, PAD_ID, UNK_ID
from babi_tools.ood_augmentation import DEFAULT_CONFIG_FILE

tf.set_random_seed(273)

with open(DEFAULT_CONFIG_FILE) as babi_config_in:
    BABI_CONFIG = json.load(babi_config_in)


def main(in_clean_dataset_folder, in_noisy_dataset_folder, in_model_folder):
    rev_vocab, kb, action_templates, config = load_model(in_model_folder)
    clean_dialogs, clean_indices = read_dialogs(os.path.join(in_clean_dataset_folder, 'dialog-babi-task6-dstc2-tst.txt'),
                                                with_indices=True)
    noisy_dialogs, noisy_indices = read_dialogs(os.path.join(in_noisy_dataset_folder, 'dialog-babi-task6-dstc2-tst.txt'),
                                                with_indices=True)

    max_noisy_dialog_length = max([item['end'] - item['start'] + 1 for item in noisy_indices])
    config['max_input_length'] = max_noisy_dialog_length
    post_ood_turns_clean, post_ood_turns_noisy = mark_post_ood_turns(noisy_dialogs, BABI_CONFIG['backoff_utterance'].lower())

    et = EntityTracker(kb)
    at = ActionTracker(None, et)
    at.set_action_templates(action_templates)

    vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    data_clean = make_dataset_for_vhcn_v2(clean_dialogs, clean_indices, vocab, et, at, **config)
    data_noisy = make_dataset_for_vhcn_v2(noisy_dialogs, noisy_indices, vocab, et, at, **config)

    context_features_clean, action_masks_clean = data_clean[2:4]
    net = DialogLevelLSTMv2(config, context_features_clean.shape[-1], action_masks_clean.shape[-1], vocab)
    net.restore(in_model_folder)

    eval_stats_clean = evaluate_advanced(net,
                                         data_clean,
                                         at.action_templates,
                                         BABI_CONFIG['backoff_utterance'].lower(),
                                         post_ood_turns=post_ood_turns_clean,
                                         runs_number=1)
    print('Clean dataset: {} turns overall'.format(eval_stats_clean['total_turns']))
    print('Accuracy:')
    accuracy = eval_stats_clean['correct_turns'] / eval_stats_clean['total_turns']
    accuracy_continuous = eval_stats_clean['correct_continuous_turns'] / eval_stats_clean['total_turns']
    accuracy_post_ood = eval_stats_clean['correct_post_ood_turns'] / eval_stats_clean['total_post_ood_turns'] \
        if eval_stats_clean['total_post_ood_turns'] != 0 \
        else 0
    print('overall: {:.3f}; continuous: {:.3f}; directly post-OOD: {:.3f}'.format(accuracy, accuracy_continuous, accuracy_post_ood))
    print('Loss : {:.3f}'.format(eval_stats_clean['avg_loss']))
    eval_stats_noisy = evaluate_advanced(net,
                                         data_noisy, 
                                         at.action_templates,
                                         BABI_CONFIG['backoff_utterance'].lower(),
                                         post_ood_turns=post_ood_turns_noisy,
                                         runs_number=1)
    print('\n\n')
    print('Noisy dataset: {} turns overall'.format(eval_stats_noisy['total_turns']))
    print('Accuracy:')
    accuracy = eval_stats_noisy['correct_turns'] / eval_stats_noisy['total_turns']
    accuracy_continuous = eval_stats_noisy['correct_continuous_turns'] / eval_stats_noisy['total_turns']
    accuracy_post_ood = eval_stats_noisy['correct_post_ood_turns'] / eval_stats_noisy['total_post_ood_turns'] \
        if eval_stats_noisy['total_post_ood_turns'] != 0 \
        else 0
    accuracy_ood = eval_stats_noisy['correct_ood_turns'] / eval_stats_noisy['total_ood_turns'] \
        if eval_stats_noisy['total_ood_turns'] != 0 \
        else 0
    print('overall: {:.3f}; continuous: {:.3f}; directly post-OOD: {:.3f}; OOD: {:.3f}'.format(accuracy,
                                                                                               accuracy_continuous,
                                                                                               accuracy_post_ood,
                                                                                               accuracy_ood))
    print('Loss : {:.3f}'.format(eval_stats_noisy['avg_loss']))


def configure_argument_parser():
    result_parser = ArgumentParser('Evaluate a trained Hybrid Code Network on bAbI Dialog Tasks data')
    result_parser.add_argument('clean_dataset_folder')
    result_parser.add_argument('noisy_dataset_folder')
    result_parser.add_argument('model_folder')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.clean_dataset_folder, args.noisy_dataset_folder, args.model_folder)

