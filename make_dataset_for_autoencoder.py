import os
import random
from argparse import ArgumentParser
from operator import itemgetter

import numpy as np
import pandas as pd

from babi_tools.lib.babi import load_dataset 
from babi_tools.ood_augmentation import load_ood

random.seed(273)
np.random.seed(273)


def configure_argument_parser():
    result_parser = ArgumentParser('Autoencoder data generation from a bAbI Dialog Task dataset')
    result_parser.add_argument('babi_folder')
    result_parser.add_argument('result_folder')
    return result_parser


def save_txt(in_lines, in_dst_file_name):
    with open(in_dst_file_name, 'w', encoding='utf-8') as lines_out:
        for line in in_lines:
            print(line, file=lines_out)


def extract_agent_utterances(in_dialogues, in_agent_name):
    utterances = []
    for dialogue_name, dialogue in in_dialogues:
        utterances += list(map(itemgetter('text'), filter(lambda turn: turn.get('agent') == in_agent_name, dialogue)))
    return utterances


def main(in_babi_folder, in_result_folder):
    dialogues = load_dataset(in_babi_folder, 'task6-dstc2')
    train, dev, test = list(map(lambda x: extract_agent_utterances(x, 'user'),
                                dialogues))
    if not os.path.exists(in_result_folder):
        os.makedirs(in_result_folder)

    ood = []
    for key, values in load_ood().items():
        ood += values
    unique_test = list(set(test))
    unique_ood = list(set(ood))
    eval_set = np.concatenate([unique_test,
                               np.random.choice(unique_ood, size=len(unique_test))])
    eval_df = pd.DataFrame({'utterance': eval_set,
                            'label': np.concatenate([np.ones([len(unique_test)], dtype=np.int32),
                                                     np.zeros([len(unique_test)], dtype=np.int32)])})
    eval_df.to_json(os.path.join(in_result_folder, 'evalset.json'))
    save_txt(train, os.path.join(in_result_folder, 'trainset.txt'))
    save_txt(dev, os.path.join(in_result_folder, 'devset.txt'))
    save_txt(test, os.path.join(in_result_folder, 'testset.txt'))


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    main(args.babi_folder, args.result_folder)
