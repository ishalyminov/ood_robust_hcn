from __future__ import print_function

import json
from collections import defaultdict

import numpy as np

from .evaluation import evaluate, evaluate_advanced
from utils.training_utils import batch_generator
from utils.preprocessing import PAD, UNK
from babi_tools.ood_augmentation import DEFAULT_CONFIG_FILE

with open(DEFAULT_CONFIG_FILE) as babi_config_in:
    BABI_CONFIG = json.load(babi_config_in)


class Trainer(object):
    def __init__(self,
                 in_train,
                 in_dev,
                 in_test,
                 in_action_templates,
                 in_random_input,
                 in_post_ood_turns_noisy,
                 in_config,
                 in_model,
                 in_model_folder):
        self.config = in_config
        self.model_folder = in_model_folder
        self.batch_size = in_config['batch_size']

        self.action_templates = in_action_templates
        self.action_weights = self.compute_dummy_action_weights(in_train[-1].flatten()) 
        action_weighting_function = np.vectorize(self.action_weights.__getitem__)
        self.data_train = in_train[:] + [action_weighting_function(in_train[-1])]
        self.data_dev = in_dev[:] + [action_weighting_function(in_dev[-1])]
        self.data_test = in_test[:] + [action_weighting_function(in_test[-1])]
        self.random_input = in_random_input
        self.post_ood_turns_noisy = in_post_ood_turns_noisy

        self.net = in_model

    def compute_action_weights(self, in_actions):
        action_counts = defaultdict(lambda: 0)
        pad_action_id = self.action_templates.index(PAD)
        unk_action_id = self.action_templates.index(UNK)
        action_counts[pad_action_id] = np.inf
        total_actions = 0
        for action in in_actions:
            if action == pad_action_id:
                continue
            action_counts[action] += 1
            total_actions += 1
        action_counts[unk_action_id] = self.config['random_input_prob'] * total_actions \
            if self.config['random_input_prob'] \
            else np.inf
        result = defaultdict(lambda: 0)
        for action, count in action_counts.items():
            result[action] = total_actions / count
        return result

    def compute_dummy_action_weights(self, in_actions):
        pad_action_id = self.action_templates.index(PAD)
        result = defaultdict(lambda: 1.0)
        result[pad_action_id] = 0.0
        return result
 
    def drop_out_batch(self, in_batch):
        X, context_features, action_masks, prev_action, ae_reconstruction_scores, y, y_weight = in_batch
        num_turns = np.sum(np.vectorize(lambda x: x!= 0)(y))
        word_dropout_prob = self.config.get('word_dropout_ratio', 0.0)
        random_input_prob = self.config.get('random_input_prob', 0.0)
        unk_action_id = len(self.action_templates) - 1 
        ae_fake_score_lower, ae_fake_score_upper = self.config['alpha'], self.config['beta']
        for idx in range(num_turns):
            for i in range(self.config['max_sequence_length']):
                if X[0][idx][i] == 0:
                    continue
                if np.random.random() < word_dropout_prob:
                    X[0][idx][i] = np.random.choice(range(4, self.config['vocabulary_size']))
        for idx in range(num_turns - 1, 0, -1):
            if np.random.random() < random_input_prob:
                random_input_idx = np.random.choice(range(self.random_input[0].shape[0]))
                random_input = [random_input_i[random_input_idx] for random_input_i in self.random_input]
                X[0][idx] = random_input[0]
                y[0][idx] = unk_action_id
                ae_reconstruction_scores[0][idx] = \
                    ae_fake_score_lower + (ae_fake_score_upper - ae_fake_score_lower) * np.random.random()
                y_weight[0][idx] = self.action_weights[unk_action_id]
                if idx + 1 < num_turns:
                    prev_action[0][idx + 1] = unk_action_id
        return in_batch

    def train(self):
        print('\n:: training started\n')
        epochs = self.config['epochs']
        best_dev_accuracy = 0.0
        epochs_without_improvement = 0
        best_eval_result = None
        for j in range(epochs):
            batch_gen = batch_generator(self.data_train, self.batch_size)
            for batch in batch_gen:
                batch_copy = [np.copy(elem) for elem in batch]
                dropped_out_batch = self.drop_out_batch(batch_copy)
                batch_loss_dict, lr = self.net.train_step(*dropped_out_batch)
            # evaluate every epoch
            train_accuracy, train_loss_dict = evaluate(self.net, self.data_train)
            train_loss_report = ' '.join(['{}: {:.3f}'.format(key, value) for key, value in train_loss_dict.items()])
            dev_accuracy, dev_loss_dict = evaluate(self.net, self.data_dev, runs_number=3)
            dev_loss_report = ' '.join(['{}: {:.3f}'.format(key, value) for key, value in dev_loss_dict.items()])
            print(':: {}@lr={:.5f} || trn accuracy {:.3f} {} || dev accuracy {:.3f} {}'.format(j + 1, lr, train_accuracy, train_loss_report, dev_accuracy, dev_loss_report))

            eval_stats_noisy = evaluate_advanced(self.net,
                                                 self.data_test,
                                                 self.action_templates,
                                                 BABI_CONFIG['backoff_utterance'],
                                                 post_ood_turns=self.post_ood_turns_noisy,
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
            if best_dev_accuracy < dev_accuracy:
                print('New best dev loss. Saving checkpoint')
                self.net.save(self.model_folder)
                best_dev_accuracy = dev_accuracy
                best_eval_result = eval_stats_noisy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if self.config['early_stopping_threshold'] < epochs_without_improvement:
                print('Finished after {} epochs due to early stopping'.format(j))
                break
        print(best_dev_accuracy)
        return best_eval_result

