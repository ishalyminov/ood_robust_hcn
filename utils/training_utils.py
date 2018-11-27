from __future__ import print_function

import json
import os


def batch_generator(data, batch_size, verbose=True, batch_limit=None):
    """
        Args:
          data : a list of nd-arrays with the same number of rows (=data points),
                 e.g. [X, y, class_weight, loss_mask]
          batch_size : an integer
    """
    batch_start_idx = 0
    total_batches_number = int(data[0].shape[0] / batch_size)
    batch_counter = 0
    while batch_start_idx < data[0].shape[0] and (not batch_limit or batch_counter < batch_limit):
        if verbose and batch_counter % 1 == 0:
            print('\rProcessed {} out of {} batches'.format(batch_counter, total_batches_number), end='')
        batch = [data_i[batch_start_idx: batch_start_idx + batch_size] for data_i in data]
        batch_start_idx += batch_size
        batch_counter += 1
        yield batch
    print('\r', end='')


def save_model(in_vocab, in_config, in_model_folder):
    if not os.path.exists(in_model_folder):
        os.makedirs(in_model_folder)
    with open(os.path.join(in_model_folder, 'config.json'), 'w', encoding='utf-8') as config_out:
        json.dump(in_config, config_out)
    with open(os.path.join(in_model_folder, 'vocab.txt'), 'w', encoding='utf-8') as vocab_out:
        print('\n'.join(in_vocab), file=vocab_out)


def load_model(in_model_folder):
    with open(os.path.join(in_model_folder, 'config.json'), encoding='utf-8') as config_in:
        config = json.load(config_in)
    with open(os.path.join(in_model_folder, 'vocab.txt'), encoding='utf-8') as vocab_in:
        vocab = [line.rstrip() for line in vocab_in]
    return vocab, config
