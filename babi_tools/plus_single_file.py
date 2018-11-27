from __future__ import print_function

import json
from argparse import ArgumentParser
from itertools import cycle
from os import path, makedirs
import random
from collections import defaultdict

import numpy as np

from lib.babi import extract_slot_values, get_files_list, read_task
from lib.babi_plus import get_enclosing_phrase

random.seed(273)
np.random.seed(273)

DEFAULT_CONFIG_FILE = 'babi_plus.json'
CONFIG = None

ACTION_LIST = None
STATS = defaultdict(lambda: 0)


def perform_action(in_action, in_dialog, in_token_coordinates, in_slot_values):
    utterance_index, token_index = in_token_coordinates
    word = in_dialog[utterance_index]['text'][token_index]
    templates = CONFIG['action_templates'][in_action]
    action_outcome = None if not len(templates) else np.random.choice(templates)
    if in_action == 'correct':
        if word in in_slot_values:
            replacement_map = {
                '$incorrect_value': np.random.choice([
                    value
                    for value in in_slot_values
                    if value != word
                ]),
                '$correct_value': word
            }
            in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
                action_outcome,
                replacement_map
            ).split()
    # can only be taken at the PP start
    if in_action == 'pp_restart':
        pp = word
        if in_dialog[utterance_index]['pos'][token_index + 1] == 'DT':
            pp += ' ' + in_dialog[utterance_index]['text'][token_index + 1]
        replacement_map = {
            '$token': word,
            '$pp': pp
        }
        in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
                action_outcome,
                replacement_map
        ).split()
    if in_action == 'correct_long_distance':
        phrase_begin, phrase_end = get_enclosing_phrase(
            in_dialog[utterance_index]['text'],
            token_index
        )
        if word in in_slot_values:
            correct_phrase = ' '.join(
                in_dialog[utterance_index]['text'][phrase_begin: phrase_end + 1]
            )
            incorrect_phrase = in_dialog[utterance_index]['text'][phrase_begin: phrase_end + 1]
            incorrect_phrase[token_index - phrase_begin] = np.random.choice([
                value
                for value in in_slot_values
                if value != word
            ])
            incorrect_phrase = ' '.join(incorrect_phrase)
            replacement_map = {
                '$incorrect_phrase': incorrect_phrase,
                '$correct_phrase': correct_phrase
            }
            in_dialog[utterance_index]['text'][phrase_begin:phrase_end + 1] = apply_replacements(
                action_outcome,
                replacement_map
            ).split()
    if in_action == 'multiturn_correct':
        if word in in_slot_values:
            replacement_map = {
                '$incorrect_value': np.random.choice([
                    value
                    for value in in_slot_values
                    if value != word
                ]),
                '$correct_value': word
            }
            in_dialog[utterance_index]['text'][token_index] = replacement_map['$incorrect_value']
            correction_turn = {
                'agent': 'usr',
                'text': apply_replacements(action_outcome, replacement_map).split()
            }
            in_dialog[utterance_index + 1: utterance_index + 2] = \
                [dict(in_dialog[utterance_index + 1]), correction_turn, dict(in_dialog[utterance_index + 1])]
    if in_action == 'selfcheck' and word in in_slot_values:
            replacement_map = {'$token' : word}
            in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
                action_outcome,
                replacement_map
            ).split()
    if in_action == 'hesitate':
        replacement_map = {'$token': word}
        in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
            action_outcome,
            replacement_map
        ).split()
        in_dialog[utterance_index]['tags'][token_index:token_index + 1] = ['<e/>' '<f/>']
    if in_action == 'restart':
        replacement_map = {
            '$token': word,
            '$utterance_from_beginning': ' '.join(in_dialog[utterance_index]['text'][:token_index + 1])
        }
        in_dialog[utterance_index]['text'][token_index:token_index + 1] = apply_replacements(
            action_outcome,
            replacement_map
        ).split()


def fix_data(in_utterance):
    REPLACEMENTS = [
        # ('are looking', 'are you looking')
    ]
    for pattern, replacement in REPLACEMENTS:
        in_utterance = in_utterance.replace(pattern, replacement)
    return in_utterance


def calculate_action_probabilities(
    in_action_weights,
    in_action_weight_mask,
    in_action_limits
):
    limits = dict(in_action_limits)
    for action in limits:
        limits[action] = float(0.0 < limits[action])
    action_weights_masked = defaultdict(lambda: {})
    # action weight masks differ
    # for the cases of background words and slot values
    for case, mask_map in in_action_weight_mask.iteritems():
        sum_masked_weight = 0.0
        for action, mask_value in mask_map.iteritems():
            masked_weight = in_action_weights[action] * mask_value * limits[action]
            if action != 'NULL':
                masked_weight *= limits['GLOBAL']
            action_weights_masked[case][action] = masked_weight
            sum_masked_weight += masked_weight
        for action in action_weights_masked[case]:
            action_weights_masked[case][action] /= sum_masked_weight
        assert abs(sum(action_weights_masked[case].values()) - 1.0) < 1e-7
    return {
        case: [weight_map[action] for action in ACTION_LIST]
        for case, weight_map in action_weights_masked.iteritems()
    }


def init(in_config_file):
    global CONFIG, ACTION_LIST
    with open(in_config_file) as actions_in:
        CONFIG = json.load(actions_in)
    ACTION_LIST = sorted(CONFIG['action_templates'].keys())


def sample_transformations(in_utterance, in_slot_values):
    action_limits = dict(CONFIG['action_limits'])

    token_types = []
    for idx, (token, tag) in enumerate(zip(in_utterance['text'], in_utterance['pos'])):
        if token in slot_values:
            token_types.append('slot_value')
        elif tag == 'IN':
            token_types.append('pp_start')
        else:
            token_types.append('background_word')

    per_token_actions = []
    for token_type in token_types:
        action_probs = calculate_action_probabilities(
            CONFIG['action_weights'],
            CONFIG['action_weight_mask'],
            action_limits
        )
        action = np.random.choice(ACTION_LIST, p=action_probs[token_type])
        per_token_actions.append(action)
        action_limits[action] -= 1
        action_limits['GLOBAL'] -= 1 * int(action != 'NULL')

    count_map = defaultdict(lambda: 0)
    for action in per_token_actions:
        count_map[action] += 1
    for action, count in count_map.iteritems():
        assert count <= CONFIG['action_limits'][action]
    return per_token_actions


def augment_dialogue(in_dialogue, in_slot_values):
    slot_values_flat = reduce(lambda x, y: x + list(y), in_slot_values, [])
    dialogue_name, dialogue = in_dialogue
    tokenized_dialogue = []
    for utterance in dialogue:
        utterance['tags'] = ['<f/>' for _ in xrange(len(utterance['text']))]
        tokenized_utterance = dict(utterance)
        tokenized_utterance['text'] = fix_data(utterance['text']).split()
        tokenized_dialogue.append(tokenized_utterance)

    dialogue_modified = False
    utterances_modified = 0
    action_stats = defaultdict(lambda: 0)
    for utterance_index in xrange(len(tokenized_dialogue) - 1, -1, -1):
        utterance = tokenized_dialogue[utterance_index]
        if utterance_index % 2 == 1 or utterance['text'] == [u'<SILENCE>']:
            continue
        transformations = sample_transformations(utterance, slot_values_flat)
        if set(transformations) != {'NULL'}:
            utterances_modified += 1
        for transformation in transformations:
            action_stats[transformation] += 1

        for reverse_token_index, action in enumerate(transformations[::-1]):
            if action != 'NULL':
                dialogue_modified = True
            token_index = len(transformations) - reverse_token_index - 1
            perform_action(
                action,
                tokenized_dialogue,
                [utterance_index, token_index],
                set(
                    reduce(
                        lambda x, y: x + list(y),
                        [
                            values_set
                            for values_set in in_slot_values
                            if utterance['text'][token_index] in values_set],
                        []
                    )
                )
            )
    for utterance in tokenized_dialogue:
        utterance['text'] = ' '.join(utterance['text'])

    global STATS
    STATS['dialogues_modified'] += int(dialogue_modified)
    STATS['utterances_modified'] += utterances_modified
    for action, count in action_stats.iteritems():
        STATS[action] += count

    return tokenized_dialogue


def plus_dataset(in_src_root, in_result_size):
    dataset_files = get_files_list(in_src_root, 'task1-API-calls')
    babi_files = [(filename, read_task(filename)) for filename in dataset_files]
    full_babi = reduce(
        lambda x, y: x + y[1],
        babi_files,
        []
    )
    slots_map = extract_slot_values(full_babi)
    babi_plus = defaultdict(lambda: [])
    result_size = in_result_size if in_result_size else len(babi_files)
    for task_name, task in babi_files:
        for dialogue_index, dialogue in zip(xrange(result_size), cycle(task)):
            babi_plus[task_name].append(
                augment_dialogue(dialogue, slots_map.values())
            )
    return babi_plus


def plus_single_task(in_task, slot_values):
    slots_map = extract_slot_values(in_task) \
        if slot_values is None \
        else slot_values
    babi_plus = map(
        lambda dialogue: augment_dialogue(dialogue, slots_map.values()),
        in_task
    )
    return babi_plus


def make_dialogue_tsv(in_dialogue):
    if not len(in_dialogue):
        return ''
    first_agent = in_dialogue[0]['agent']
    utterance_idx = 0
    turn_idx = 1
    result_lines = []
    while utterance_idx < len(in_dialogue):
        if utterance_idx + 1 < len(in_dialogue) and in_dialogue[utterance_idx]['agent'] == first_agent:
            usr, sys = in_dialogue[utterance_idx: utterance_idx + 2]
            result_lines.append('{} {}\t{}'.format(turn_idx, usr['text'], sys['text']))
            utterance_idx += 2
        else:
            sys = in_dialogue[utterance_idx]
            result_lines.append('{} {}'.format(turn_idx, sys['text']))
            utterance_idx += 1
        turn_idx += 1
    return '\n'.join(result_lines)
            


def save_babble(in_dialogues, in_dst_root):
    if not path.exists(in_dst_root):
        makedirs(in_dst_root)

    for dialogue_index, dialogue in enumerate(in_dialogues):
        with open(path.join(in_dst_root, 'babi_plus_{}.txt'.format(dialogue_index)), 'w') as dialogue_out:
            print('\n'.join(['{}:\t{}'.format(utterance['agent'], utterance['text'])
                             for utterance in dialogue]),
                  file=dialogue_out)


def print_stats():
    print('Data modification statistics:')
    for key, value in STATS.iteritems():
        print('{}\t{}'.format(key, value))


def save_babi(in_dialogues, in_dst_root):
    if not path.exists(in_dst_root):
        makedirs(in_dst_root)

    for task_name, task_dialogues in in_dialogues.iteritems():
        filename = path.join(in_dst_root, path.basename(task_name))
        with open(filename, 'w') as task_out:
            for dialogue in task_dialogues:
                print(make_dialogue_tsv(dialogue) + '\n\n', file=task_out)


def configure_argument_parser():
    parser = ArgumentParser(description='generate bAbI+ data')
    parser.add_argument('babi_file', help='file with bAbI Dialogs')
    parser.add_argument('babi_plus_root', help='output folder')
    parser.add_argument(
        '--output_format',
        default='babi',
        help='format of output dialogues [babi/babble]'
    )
    parser.add_argument(
        '--result_size',
        type=int,
        default=None,
        help='size of generated dataset [default=input dataset size]'
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_FILE,
        help='dicustom disfluency config (json file)'
    )

    return parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    init(args.config)
    task = read_task(args.babi_file)
    slot_values = extract_slot_values(task)
    task_name = path.basename(args.babi_file)
    babi_plus_dialogues = plus_single_task(task, slot_values)
    save_function = locals()['save_' + args.output_format]
    save_function({task_name: babi_plus_dialogues}, args.babi_plus_root)
    print_stats()

