from __future__ import print_function

import json
import os
from argparse import ArgumentParser
import random
import sys
import copy

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import modules
from modules.entities import EntityTracker
from modules.util import get_utterances, save_model
from modules.kb import make_augmented_knowledge_base
from modules.trainer import Trainer
from modules.evaluation import mark_post_ood_turns
from utils.preprocessing import make_vocabulary, load_hcn_json, PAD, START, UNK, EOS
import utils.preprocessing
from babi_tools.ood_augmentation import DEFAULT_CONFIG_FILE

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)

with open(DEFAULT_CONFIG_FILE) as babi_config_in:
    BABI_CONFIG = json.load(babi_config_in)
BABI_FOLDER = os.path.join(os.path.dirname(__file__), 'data')


def augment_dataset_with_ae_reconstruction_scores(in_dataset, in_ae_score_map):
    result = copy.deepcopy(in_dataset)
    for dialogue in result['dialogs']:
        for utterance in dialogue['turns']:
            ae_score = in_ae_score_map.get(' '.join(utterance['input'].split()), None)
            if ae_score is None:
                print('No reconstruction score for {}'.format(utterance['input']))
            utterance['ae_reconstruction_score'] = ae_score
    return result


def extract_alpha_beta_scores(in_ae_score_map, in_config):
    alpha = np.percentile(list(in_ae_score_map.values()), 99)
    assert alpha < in_config['beta'], 'beta reconstruction score parameter must be higher that {:.3f}'.format(alpha)
    return alpha, in_config['beta']


def main(in_dataset_folder, in_noisy_dataset_folder, in_custom_vocab_file, in_model_folder, in_config):
    with open(in_config, encoding='utf-8') as config_in:
        config = json.load(config_in)
    train_json = load_hcn_json(os.path.join(in_dataset_folder, 'train.json'))
    with open(os.path.join(in_dataset_folder, 'train_ae_reconstruction_scores.json')) as train_ae_in:
        train_ae_scores_json = json.load(train_ae_in)
    train_json = augment_dataset_with_ae_reconstruction_scores(train_json, train_ae_scores_json)
    dev_json = load_hcn_json(os.path.join(in_dataset_folder, 'dev.json'))
    with open(os.path.join(in_dataset_folder, 'dev_ae_reconstruction_scores.json')) as dev_ae_in:
        dev_ae_scores_json = json.load(dev_ae_in)
    dev_json = augment_dataset_with_ae_reconstruction_scores(dev_json, dev_ae_scores_json)
    # test_json = load_hcn_json(os.path.join(in_dataset_folder, 'test.json'))
    test_ood_json = load_hcn_json(os.path.join(in_noisy_dataset_folder, 'test_ood.json'))
    with open(os.path.join(in_noisy_dataset_folder, 'test_ood_ae_reconstruction_scores.json')) as test_ae_in:
        test_ood_ae_scores_json = json.load(test_ae_in)
    test_ood_json = augment_dataset_with_ae_reconstruction_scores(test_ood_json, test_ood_ae_scores_json)

    kb = make_augmented_knowledge_base(os.path.join(BABI_FOLDER, 'dialog-babi-task6-dstc2-kb.txt'),
                                       os.path.join(BABI_FOLDER, 'dialog-babi-task6-dstc2-candidates.txt'))
    action_templates = train_json['actions']
    max_noisy_dialog_length = max([len(dialog['turns']) for dialog in test_ood_json['dialogs']])
    config['max_input_length'] = max_noisy_dialog_length

    et = EntityTracker(kb)

    post_ood_turns_clean, post_ood_turns_noisy = mark_post_ood_turns(test_ood_json)

    if in_custom_vocab_file is not None:
        with open(in_custom_vocab_file) as vocab_in:
            rev_vocab = [line.rstrip() for line in vocab_in]
            vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    else:
        utterances_tokenized = []
        for dialog in train_json['dialogs']:
            for utterance in dialog['turns']:
                utterances_tokenized.append(utterance['input'].split())

        vocab, rev_vocab = make_vocabulary(utterances_tokenized,
                                           config['max_vocabulary_size'],
                                           special_tokens=[PAD, START, UNK, EOS] + list(kb.keys()))
    ctx_features = []
    for dialog in train_json['dialogs']:
        for utterance in dialog['turns']:
            if 'context_features' in utterance:
                ctx_features.append(utterance['context_features'])
    ctx_features_vocab, ctx_features_rev_vocab = make_vocabulary(ctx_features,
                                                                 config['max_vocabulary_size'],
                                                                 special_tokens=[])
    config['vocabulary_size'] = len(vocab)

    print('Training with config: {}'.format(json.dumps(config)))
    data_preparation_function = getattr(utils.preprocessing, config['data_preparation_function'])

    data_train = data_preparation_function(train_json, vocab, ctx_features_vocab, et, **config)
    data_dev = data_preparation_function(dev_json, vocab, ctx_features_vocab, et, **config)
    # data_test = data_preparation_function(test_json, vocab, ctx_features_vocab, et, **config)
    data_test_ood = data_preparation_function(test_ood_json, vocab, ctx_features_vocab, et, **config)

    dropout_turn_generation_function = getattr(utils.preprocessing,
                                               config['dropout_turn_generation_function'])
    alpha, beta = extract_alpha_beta_scores(train_ae_scores_json, config)
    config['alpha'] = alpha
    random_input = dropout_turn_generation_function(10000,
                                                    3,
                                                    config['max_sequence_length'],
                                                    train_json,
                                                    vocab,
                                                    config)

    save_model(rev_vocab, config, kb, action_templates, in_model_folder)
    net = getattr(modules, config['model_name'])(vocab,
                                                 config,
                                                 len(ctx_features_vocab),
                                                 len(action_templates))
    trainer = Trainer(data_train,
                      data_dev,
                      data_test_ood,
                      action_templates,
                      random_input,
                      post_ood_turns_noisy,
                      config,
                      net,
                      in_model_folder)
    trainer.train()


def configure_argument_parser():
    result_parser = ArgumentParser('Train a Hybrid Code Network on bAbI Dialog Tasks data')
    result_parser.add_argument('dataset_folder')
    result_parser.add_argument('noisy_dataset_folder')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('config')
    result_parser.add_argument('--custom_vocab', default=None, type=str)
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder,
         args.noisy_dataset_folder,
         args.custom_vocab,
         args.model_folder,
         args.config)
