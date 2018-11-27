from __future__ import print_function

import json
import os
from argparse import ArgumentParser
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.entities import EntityTracker
from modules.dual_encoder_variational_hierarchical_lstm import DualEncoderVariationalHierarchicalLSTM
from modules.actions import ActionTracker
from modules.kb import make_augmented_knowledge_base
from modules.util import read_dialogs, get_utterances, save_model
from modules.evaluation import evaluate
from utils.training_utils import batch_generator
from utils.preprocessing import (make_vocabulary,
                                 make_dataset_for_variational_hcn,
                                 generate_dropout_turns_for_variational_hcn,
                                 PAD,
                                 START,
                                 UNK,
                                 EOS)

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)


class Trainer(object):
    def __init__(self, in_train, in_dev, in_test, in_action_templates, in_random_input, in_rev_vocab, in_config, in_model_folder):
        self.config = in_config
        self.model_folder = in_model_folder
        self.batch_size = in_config['batch_size']
        self.action_templates = in_action_templates

        self.data_train = in_train
        self.data_dev = in_dev
        self.random_input = in_random_input
        context_features, action_masks = self.data_train[4:6]
        self.net = DualEncoderVariationalHierarchicalLSTM(in_rev_vocab, self.config, context_features.shape[-1], action_masks.shape[-1])

    def train(self):
        print('\n:: training started\n')
        epochs = self.config['epochs']
        best_dev_accuracy = 0.0
        epochs_without_improvement = 0
        random_input_prob = self.config.get('random_input_prob', 0.0)
        unk_action_id = self.action_templates.index(UNK)
        for j in range(epochs):
            batch_gen = batch_generator(self.data_train, self.batch_size)
            for batch in batch_gen:
                batch_copy = [np.copy(elem) for elem in batch]
                enc_inp, dec_inp, dec_out, bow_out, context_features, action_masks, y = batch_copy
                num_turns = np.sum(np.vectorize(lambda x: x!= 0)(y)) 
                for idx in range(num_turns):
                    if np.random.random() < random_input_prob:
                        random_input_idx = np.random.choice(range(self.random_input[0].shape[0]))
                        random_input = [random_input_i[random_input_idx] for random_input_i in self.random_input]
                        enc_inp[0][idx], dec_inp[0][idx], dec_out[0][idx], bow_out[0][idx] = random_input 
                        y[0][idx] = unk_action_id
                batch_loss_dict, lr = self.net.train_step(enc_inp, dec_inp, dec_out, bow_out, context_features, action_masks, y)
            # evaluate every epoch
            train_accuracy, train_mean_losses = evaluate(self.net, self.data_train, runs_number=1)
            train_loss_report = ' '.join(['{}: {:.3f}'.format(key, value) for key, value in train_mean_losses.items()])
            dev_accuracy, dev_mean_losses = evaluate(self.net, self.data_dev, runs_number=1)
            dev_loss_report = ' '.join(['{}: {:.3f}'.format(key, value) for key, value in dev_mean_losses.items()])
            print(':: {}@lr={:.5f} || trn accuracy {:.3f} {} || dev accuracy {:.3f} {}'.format(j + 1, lr, train_accuracy, train_loss_report, dev_accuracy, dev_loss_report))

            if best_dev_accuracy < dev_accuracy:
                print('New best dev accuracy. Saving checkpoint')
                self.net.save(self.model_folder)
                best_dev_accuracy = dev_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if self.config['early_stopping_threshold'] < epochs_without_improvement:
                print('Finished after {} epochs due to early stopping'.format(j))
                break


def main(in_dataset_folder, in_custom_vocab_file, in_model_folder, in_config):
    with open(in_config, encoding='utf-8') as config_in:
        config = json.load(config_in)

    train_dialogs, train_indices = read_dialogs(os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-trn.txt'),
                                                with_indices=True)
    dev_dialogs, dev_indices = read_dialogs(os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-dev.txt'),
                                            with_indices=True)
    test_dialogs, test_indices = read_dialogs(os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-tst.txt'),
                                              with_indices=True)

    kb = make_augmented_knowledge_base(os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-kb.txt'),
                                       os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-candidates.txt'))

    et = EntityTracker(kb)
    at = ActionTracker(os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-candidates.txt'), et)

    if in_custom_vocab_file is not None:
        with open(in_custom_vocab_file) as vocab_in:
            rev_vocab = [line.rstrip() for line in vocab_in]
            vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    else:
        utterances_tokenized = []
        for dialog in train_dialogs:
            utterances_tokenized += list(map(lambda x: x.split(), dialog))

        vocab, rev_vocab = make_vocabulary(utterances_tokenized,
                                           config['max_vocabulary_size'],
                                           special_tokens=[PAD, START, UNK, EOS] + list(kb.keys()))
    config['vocabulary_size'] = len(vocab)

    data_train = make_dataset_for_variational_hcn(train_dialogs, train_indices, vocab, et, at, **config)
    data_dev = make_dataset_for_variational_hcn(dev_dialogs, dev_indices, vocab, et, at, **config)
    data_test = make_dataset_for_variational_hcn(test_dialogs, test_indices, vocab, et, at, **config)

    random_input = generate_dropout_turns_for_variational_hcn(10000,
                                                              config['max_sequence_length'],
                                                              [utterance[0] for utterance in train_dialogs],
                                                              vocab,
                                                              config['turn_word_dropout_prob'])

    save_model(rev_vocab, config, kb, at.action_templates, in_model_folder)
    trainer = Trainer(data_train,
                      data_dev,
                      data_test,
                      at.action_templates,
                      random_input,
                      rev_vocab,
                      config,
                      in_model_folder)
    trainer.train()


def configure_argument_parser():
    result_parser = ArgumentParser('Train a Variational Hybrid Code Network on bAbI Dialog Tasks data')
    result_parser.add_argument('dataset_folder')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('--custom_vocab', default=None, type=str)
    result_parser.add_argument('--config', default='de_vhcn.json')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.custom_vocab, args.model_folder, args.config)

