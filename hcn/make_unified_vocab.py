from __future__ import print_function

import json
import os
from argparse import ArgumentParser

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.preprocessing import make_vocabulary, read_dialogs, PAD, START, UNK, EOS


def main(in_dataset_folder, in_result_file, in_config):
    with open(in_config, encoding='utf-8') as config_in:
        config = json.load(config_in)
    # *_ood.json files contain both IND and OOD words
    train_dialogs, train_indices = read_dialogs(os.path.join(in_dataset_folder, 'train_ood.json'))
    dev_dialogs, dev_indices = read_dialogs(os.path.join(in_dataset_folder, 'dev_ood.json'))
    test_dialogs, test_indices = read_dialogs(os.path.join(in_dataset_folder, 'test_ood.json'))

    utterances_tokenized = []
    for input_utterance, action_id in train_dialogs + dev_dialogs + test_dialogs:
        utterances_tokenized.append(input_utterance.split())
    print(utterances_tokenized[:10])
    vocab, rev_vocab = make_vocabulary(utterances_tokenized,
                                       config['max_vocabulary_size'],
                                       special_tokens=[PAD, START, UNK, EOS])
    with open(in_result_file, 'w') as vocab_out:
        for word in rev_vocab:
            print(word, file=vocab_out)


def configure_argument_parser():
    result_parser = ArgumentParser('Make a unified vocabulary out of all the dataset\'s files')
    result_parser.add_argument('dataset_folder')
    result_parser.add_argument('result_file')
    result_parser.add_argument('--config', default='hcn.json')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.result_file, args.config)
