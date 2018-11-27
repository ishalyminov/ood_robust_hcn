from __future__ import print_function

import os
from argparse import ArgumentParser
import random
import sys
import json
from collections import defaultdict

import numpy as np

random.seed(273)
np.random.seed(273)

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.entities import EntityTracker
from modules.util import read_dialogs, get_utterances, load_model
from modules.evaluation import evaluate, evaluate_advanced, mark_post_ood_turns
import modules
from utils.preprocessing import make_vocabulary, load_hcn_json
import utils.preprocessing
from babi_tools.ood_augmentation import DEFAULT_CONFIG_FILE

tf.set_random_seed(273)

with open(DEFAULT_CONFIG_FILE) as babi_config_in:
    BABI_CONFIG = json.load(babi_config_in)
BABI_FOLDER = os.path.join(os.path.dirname(__file__), 'data')


def main(in_clean_dataset_folder, in_noisy_dataset_folder, in_model_folder, in_mode, in_runs_number):
    rev_vocab, kb, action_templates, config = load_model(in_model_folder)
    train_json = load_hcn_json(os.path.join(in_clean_dataset_folder, 'train.json'))
    test_json = load_hcn_json(os.path.join(in_clean_dataset_folder, 'test.json'))
    test_ood_json = load_hcn_json(os.path.join(in_noisy_dataset_folder, 'test_ood.json'))

    max_noisy_dialog_length = max([len(dialog['turns']) for dialog in test_ood_json['dialogs']])
    config['max_input_length'] = max_noisy_dialog_length
    post_ood_turns_clean, post_ood_turns_noisy = mark_post_ood_turns(test_ood_json)

    et = EntityTracker(kb)

    action_weights = defaultdict(lambda: 1.0)
    action_weights[0] = 0.0
    action_weighting = np.vectorize(action_weights.__getitem__)
    vocab = {word: idx for idx, word in enumerate(rev_vocab)}

    ctx_features = []
    for dialog in train_json['dialogs']:
        for utterance in dialog['turns']:
            if 'context_features' in utterance:
                ctx_features.append(utterance['context_features'])
    ctx_features_vocab, ctx_features_rev_vocab = make_vocabulary(ctx_features,
                                                                 config['max_vocabulary_size'],
                                                                 special_tokens=[])

    data_preparation_function = getattr(utils.preprocessing, config['data_preparation_function'])
    data_clean = data_preparation_function(test_json, vocab, ctx_features_vocab, et, **config)
    data_noisy = data_preparation_function(test_ood_json, vocab, ctx_features_vocab, et, **config)

    data_clean_with_weights = *data_clean, action_weighting(data_clean[-1])
    data_noisy_with_weights = *data_noisy, action_weighting(data_noisy[-1])

    net = getattr(modules, config['model_name'])(vocab,
                                                 config,
                                                 len(ctx_features_vocab),
                                                 len(action_templates))
    net.restore(in_model_folder)

    if in_mode == 'clean':
        eval_stats_clean = evaluate_advanced(net,
                                             data_clean_with_weights,
                                             action_templates,
                                             BABI_CONFIG['backoff_utterance'],
                                             post_ood_turns=post_ood_turns_clean,
                                             runs_number=in_runs_number)
        print('Clean dataset: {} turns overall'.format(eval_stats_clean['total_turns']))
        print('Accuracy:')
        accuracy = eval_stats_clean['correct_turns'] / eval_stats_clean['total_turns']
        accuracy_continuous = eval_stats_clean['correct_continuous_turns'] / eval_stats_clean['total_turns']
        accuracy_post_ood = eval_stats_clean['correct_post_ood_turns'] / eval_stats_clean['total_post_ood_turns'] \
            if eval_stats_clean['total_post_ood_turns'] != 0 \
            else 0
        ood_f1 = eval_stats_clean['ood_f1']
        print('overall acc: {:.3f}; continuous acc: {:.3f}; '
              'directly post-OOD acc: {:.3f}; OOD F1: {:.3f}'.format(accuracy,
                                                                     accuracy_continuous,
                                                                     accuracy_post_ood,
                                                                     ood_f1))
        print('Loss : {:.3f}'.format(eval_stats_clean['avg_loss']))
    elif in_mode == 'noisy':
        eval_stats_noisy = evaluate_advanced(net,
                                             data_noisy_with_weights, 
                                             action_templates,
                                             BABI_CONFIG['backoff_utterance'],
                                             post_ood_turns=post_ood_turns_noisy,
                                             runs_number=in_runs_number)
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
        ood_f1 = eval_stats_noisy['ood_f1']
        print('overall acc: {:.3f}; continuous acc: {:.3f}; '
              'directly post-OOD acc: {:.3f}; OOD acc: {:.3f}; OOD F1: {:.3f}'.format(accuracy,
                                                                                      accuracy_continuous,
                                                                                      accuracy_post_ood,
                                                                                      accuracy_ood,
                                                                                      ood_f1))
        print('Loss : {:.3f}'.format(eval_stats_noisy['avg_loss']))
    elif in_mode == 'noisy_ignore_ood':
        eval_stats_no_ood = evaluate_advanced(net,
                                              data_noisy_with_weights,
                                              action_templates,
                                              BABI_CONFIG['backoff_utterance'],
                                              post_ood_turns=post_ood_turns_noisy,
                                              ignore_ood_accuracy=True,
                                              runs_number=in_runs_number)
        print('Accuracy (OOD turns ignored):')
        accuracy = eval_stats_no_ood['correct_turns'] / eval_stats_no_ood['total_turns']
        accuracy_after_ood = eval_stats_no_ood['correct_turns_after_ood'] / eval_stats_no_ood['total_turns_after_ood'] \
            if eval_stats_no_ood['total_turns_after_ood'] != 0 \
            else 0
        accuracy_post_ood = eval_stats_no_ood['correct_post_ood_turns'] / eval_stats_no_ood['total_post_ood_turns'] \
            if eval_stats_no_ood['total_post_ood_turns'] != 0 \
            else 0
        print('overall: {:.3f}; '
              'after first OOD: {:.3f}; '
              'directly post-OOD: {:.3f}'.format(accuracy, accuracy_after_ood, accuracy_post_ood))


def configure_argument_parser():
    result_parser = ArgumentParser('Evaluate a Hybrid Code Network on bAbI Dialog Tasks data')
    result_parser.add_argument('clean_dataset_folder')
    result_parser.add_argument('noisy_dataset_folder')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('eval_mode', help='clean/noisy/noisy_ignore_ood')
    result_parser.add_argument('--runs_number', type=int, default=3)
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.clean_dataset_folder, args.noisy_dataset_folder, args.model_folder, args.eval_mode, args.runs_number)

