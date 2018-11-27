from __future__ import print_function

import json
import os
from argparse import ArgumentParser
import sys

from gensim.models import KeyedVectors

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.util import read_dialogs
from modules.kb import make_augmented_knowledge_base
from utils.preprocessing import make_vocabulary, PAD, START, UNK, EOS, UtteranceEmbed


def main(in_dataset_folder, in_result_file, in_max_size, in_config):
    with open(in_config, encoding='utf-8') as config_in:
        config = json.load(config_in)
    w2v_model = KeyedVectors.load_word2vec_format(UtteranceEmbed.DEFAULT_W2V_FILE, binary=True) 
    kb = make_augmented_knowledge_base(os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-kb.txt'),
                                       os.path.join(in_dataset_folder, 'dialog-babi-task6-dstc2-candidates.txt'))

    vocab = [PAD, START, UNK, EOS] + list(kb.keys())
    vocab += [word for word, vocab_entry in sorted(w2v_model.vocab.items(), key=lambda x: x[1].count, reverse=True)[:in_max_size - len(vocab)]]
    with open(in_result_file, 'w') as vocab_out:
        for word in vocab:
            print(word, file=vocab_out)


def configure_argument_parser():
    result_parser = ArgumentParser('Make a unified vocabulary out of all the dataset\'s files')
    result_parser.add_argument('dataset_folder')
    result_parser.add_argument('result_file')
    result_parser.add_argument('max_size', type=int)
    result_parser.add_argument('--config', default='hcn.json')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.result_file, args.max_size, args.config)

