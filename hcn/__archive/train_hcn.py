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
from modules.dialog_level_lstm import DialogLevelLSTM
from modules.actions import ActionTracker
from modules.kb import make_augmented_knowledge_base

from modules.util import read_dialogs, get_utterances, save_model
from utils.training_utils import batch_generator
from utils.preprocessing import make_vocabulary, make_dataset_for_vhcn_v2, generate_dropout_turns_for_de_vhcn_v3, PAD, UNK, START, EOS
from modules.evaluation import evaluate

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)


class Trainer(object):
    def __init__(self, in_train, in_dev, in_test, in_action_templates, in_random_input, in_config, in_vocab, in_model_folder):
        self.config = in_config
        self.model_folder = in_model_folder
        self.batch_size = in_config['batch_size']

        self.X_train, self.bow_features_train, self.context_features_train, self.action_masks_train, self.prev_actions_train, self.y_train = in_train
        self.X_dev, self.bow_features_dev, self.context_features_dev, self.action_masks_dev, self.prev_actions_dev, self.y_dev = in_dev

        self.action_templates = in_action_templates
        self.random_input = in_random_input
        self.net = DialogLevelLSTM(self.config, self.context_features_train.shape[-1], self.action_masks_train.shape[-1], in_vocab)

    def train(self):
        print('\n:: training started\n')
        epochs = self.config['epochs']
        best_accuracy = 0.0
        epochs_without_improvement = 0
        unk_action_id = self.action_templates.index(UNK)
        word_dropout_prob = self.config['word_dropout_ratio']
        random_input_prob = self.config['random_input_prob']
        for j in range(epochs):
            losses = []
            batch_gen = batch_generator([self.X_train, self.bow_features_train, self.context_features_train, self.action_masks_train, self.prev_actions_train, self.y_train], self.batch_size)
            for batch in batch_gen:
                batch_copy = [np.copy(batch_elem) for batch_elem in batch]
                batch_x, batch_bow, batch_context_features, batch_action_masks, batch_prev_action, batch_y = batch_copy
                num_turns = np.sum(np.vectorize(lambda x: x!= 0)(batch_y))
                for idx in range(num_turns):
                    for i in range(self.config['max_sequence_length']):
                        if batch_x[0][idx][i] == 0:
                            continue
                        if np.random.random() < word_dropout_prob:
                            batch_x[0][idx][i] = np.random.choice(range(4, self.config['vocabulary_size']))
                for idx in range(num_turns):
                    if np.random.random() < random_input_prob:
                        random_input_idx = np.random.choice(range(self.random_input[0].shape[0]))
                        random_input = [random_input_i[random_input_idx] for random_input_i in self.random_input]
                        batch_x[0][idx] = random_input[0]
                        batch_bow[0][idx] = random_input[1]
                        batch_y[0][idx] = unk_action_id
                        if idx + 1 < num_turns:
                            batch_prev_action[0][idx + 1] = unk_action_id  
                batch_loss_dict, lr = self.net.train_step(batch_x, batch_bow, batch_context_features, batch_action_masks, batch_prev_action, batch_y)
                losses += batch_loss_dict['loss'].tolist()

            # evaluate every epoch
            train_accuracy, train_loss = evaluate(self.net, (self.X_train, self.bow_features_train, self.context_features_train, self.action_masks_train, self.prev_actions_train, self.y_train))
            dev_accuracy, dev_loss = evaluate(self.net, (self.X_dev, self.bow_features_dev, self.context_features_dev, self.action_masks_dev, self.prev_actions_dev, self.y_dev))
            print(':: {} || trn accuracy {:.3f} loss {:.3f} || dev accuracy {:.3f} loss {:.3f}'.format(j+1, train_accuracy, train_loss['loss'], dev_accuracy, dev_loss['loss']))

            if best_accuracy < dev_accuracy:
                print('New best dev accuracy. Saving checkpoint')
                self.net.save(self.model_folder)
                best_accuracy = dev_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if self.config['early_stopping_threshold'] < epochs_without_improvement:
                print('Finished after {} epochs due to early stopping'.format(j))
                break


def main(in_dataset_folder, in_model_folder, in_config, in_custom_vocab_file):
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

    et = EntityTracker(kb)
    at = ActionTracker(os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-candidates.txt'), et)

    data_train = make_dataset_for_vhcn_v2(train_dialogs, train_indices, vocab, et, at, **config)
    data_dev = make_dataset_for_vhcn_v2(dev_dialogs, dev_indices, vocab, et, at, **config)
    data_test = make_dataset_for_vhcn_v2(test_dialogs, test_indices, vocab, et, at, **config)

    random_input = generate_dropout_turns_for_de_vhcn_v3(10000,
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
                      config,
                      vocab,
                      in_model_folder)
    trainer.train()


def configure_argument_parser():
    result_parser = ArgumentParser('Train a Hybrid Code Network on bAbI Dialog Tasks data')
    result_parser.add_argument('dataset_folder')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('--config', default='hcn.json')
    result_parser.add_argument('--custom_vocab', default=None, type=str)
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.model_folder, args.config, args.custom_vocab)

