import json
import os
import re
import sys
import random
from operator import itemgetter
from argparse import ArgumentParser
from itertools import cycle
from collections import defaultdict

import nltk

sys.path.append(os.path.dirname(__file__))

from lib.babi import load_dataset, read_task
from plus_single_file import make_dialogue_tsv

random.seed(273)


DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'ood_augmentation.json')


class FiniteStateOODInjector(object):
    def __init__(self, ood_prob, ood_continuation_prob, **kwargs):
        self.ood_prob = ood_prob
        self.ood_continuation_prob = ood_continuation_prob
        self.state = None
        self.reset_state()

    def reset_state(self):
        self.state = 'fluent'

    def sample_next_state(self):
        rnd = random.random()
        if self.state == 'fluent':
            if rnd < self.ood_prob:
                self.state = 'ood'
        elif self.state == 'ood':
            if self.ood_continuation_prob <= rnd:
                self.state = 'fluent'
        return self.state


def pos_tag(in_utterance):
    return list(map(itemgetter(1), nltk.pos_tag(in_utterance.split())))


def load_ood():
    result = defaultdict(lambda: [])
    with open(os.path.join(DATA_FOLDER, 'twitter_ood_positive.txt'), encoding='utf-8') as noise_positive_in:
        result['reactions_positive'] += [line.strip() for line in noise_positive_in.readlines()]
    with open(os.path.join(DATA_FOLDER, 'twitter_ood_negative.txt'), encoding='utf-8') as noise_negative_in:
        result['reactions_negative'] += [line.strip() for line in noise_negative_in.readlines()]
    with open(os.path.join(DATA_FOLDER, 'twitter_ood_breakdown.txt'), encoding='utf-8') as twitter_breakdown_in:
        result['breakdown'] += [line.strip() for line in twitter_breakdown_in.readlines()]
    with open(os.path.join(DATA_FOLDER, 'reddit_ood_breakdown.txt'), encoding='utf-8') as reddit_breakdown_in:
        result['breakdown'] += [line.strip() for line in reddit_breakdown_in.readlines()]
    with open(os.path.join(DATA_FOLDER, 'stanford_ood.txt'), encoding='utf-8') as stanford_in:
        result['foreign_domain'] += [line.strip() for line in stanford_in.readlines()]
    with open(os.path.join(DATA_FOLDER, 'dstc1_ood.txt'), encoding='utf-8') as dstc1_in:
        result['foreign_domain'] += [line.strip() for line in dstc1_in.readlines()]
    with open(os.path.join(DATA_FOLDER, 'frames_ood.txt'), encoding='utf-8') as frames_in:
        result['foreign_domain'] += [line.strip() for line in frames_in.readlines()]
    return result


def augment_dialogue_with_ood_turn_level(in_dialogue,
                                         in_ood,
                                         in_config,
                                         stats=defaultdict(lambda: 0)):
    # type: (list, dict, dict, dict) -> list
    injector = FiniteStateOODInjector(**in_config)
    augmented_dialogue = []
    for turn in in_dialogue:
        if turn['agent'] == 'user' and turn['text'] != '<SILENCE>':
            next_state = injector.sample_next_state()
            if next_state == 'ood':
                ood_sequence_length = 0
                while next_state == 'ood':
                    ood_sequence_length += 1
                    ood_turn = random.choice(in_ood['foreign_domain'])
                    augmented_dialogue.append({'agent': 'user', 'pos': pos_tag(ood_turn), 'text': ood_turn})
                    augmented_dialogue.append({'agent': 'system',
                                               'pos': in_config['backoff_utterance_pos'],
                                               'text': in_config['backoff_utterance']})
                    next_state = injector.sample_next_state()
                stats[ood_sequence_length] += 1
                turn['text'] = random.choice(in_ood['breakdown']) + ' ' + turn['text']
                turn['pos'] = pos_tag(turn['text'])
        augmented_dialogue.append(turn)
    if len(augmented_dialogue) != len(in_dialogue):
        stats['dialogues_augmented'] += 1
    return augmented_dialogue


def print_augmentation_stats(in_stats):
    print('OOD sequences by length:')
    for key, value in sorted(filter(lambda keyval: type(keyval[0]) == type(1), in_stats.items())):
        print('{}:\t{}'.format(key, value))
    print('Overall dialogues augmented: {}'.format(in_stats['dialogues_augmented']))


def augment_dataset(in_dialogues, in_ood, in_config, in_result_size):
    stats = defaultdict(lambda: 0) 
    augmented_dialogues = []
    for dialogue_index, (dialogue_name, dialogue) in zip(range(in_result_size), cycle(in_dialogues)):
        dialogue_name = re.sub('\.[\d]+$', '', dialogue_name)
        augmented_dialogues.append(('{}.{}'.format(dialogue_name, dialogue_index + 1),
                                    augment_dialogue_with_ood_turn_level(dialogue,
                                                                         in_ood,
                                                                         in_config,
                                                                         stats=stats)))
    print_augmentation_stats(stats)
    return augmented_dialogues


def save_babi(in_dialogues, in_dst_file):
    with open(in_dst_file, 'w', encoding='utf-8') as task_out:
        for dialogue in in_dialogues:
            print(make_dialogue_tsv(dialogue[1]) + '\n', file=task_out)


def main(in_input_file, in_result_file, in_config, in_result_size):
    ood = load_ood()
    dialogues = read_task(in_input_file)
    augmented_dialogues = augment_dataset(dialogues,
                                          ood,
                                          in_config,
                                          in_result_size if in_result_size else len(dialogues))
    save_babi(augmented_dialogues, in_result_file)


def configure_argument_parser():
    result_parser = ArgumentParser(description='Augment bAbI dialogues with out-of-domain utterances')
    result_parser.add_argument('babi_file', help='file with bAbI Dialogs')
    result_parser.add_argument('result_file', help='output file')
    result_parser.add_argument('--result_size',
                               type=int,
                               default=None,
                               help='size of generated dataset [default=input dataset size]')
    result_parser.add_argument('--config', default=DEFAULT_CONFIG_FILE)

    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    with open(args.config) as config_in:
        config = json.load(config_in)
    config['backoff_utterance_pos'] = pos_tag(config['backoff_utterance'])
    main(args.babi_file, args.result_file, config, args.result_size)
