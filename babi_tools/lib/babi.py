import os
import re
from collections import deque, defaultdict
from operator import itemgetter
from os import path
import nltk

TASK_ID = 'task1-API-calls'
DATASET_ORDERING = ['trn', 'dev', 'tst', 'tst-OOV']

nltk.download('averaged_perceptron_tagger')


def read_task(in_file_name):
    result = []
    with open(in_file_name, encoding='utf-8') as task_in:
        task_content = task_in.read()
    dialogs = [
        list(filter(lambda line: len(line.strip()), dialog.split('\n')))
        for dialog in task_content.split('\n\n')
    ]
    dialogs = list(filter(len, dialogs))

    for dialog_index, dialog in enumerate(dialogs):
        result.append(('{}.{}'.format(path.basename(in_file_name), dialog_index + 1), []))
        for line in dialog:
            line = re.sub('^\d+\s', '', line)
            turns = line.split('\t')
            if len(turns) == 1:
               system_turn = turns[0]
            if len(turns) == 2:
               user_turn, system_turn = turns
               result[-1][1].append({'agent': 'user',
                                     'text': user_turn,
                                     'pos': list(map(itemgetter(1), nltk.pos_tag(user_turn.split())))})
            result[-1][1].append({'agent': 'system',
                                  'text': system_turn,
                                  'pos': list(map(itemgetter(1), nltk.pos_tag(system_turn.split())))})
    return list(filter(lambda x: len(x[1]), result))


def preprocess_for_seq2seq(in_task_dialogs, in_config):
    encoder_input, decoder_input = [], []
    encoder_buffer = deque([], maxlen=in_config['encoder_context_size'])

    for dialog in in_task_dialogs:
        encoder_buffer.clear()
        for turn in dialog:
            user_turn, system_turn = turn['user'], turn['system']
            encoder_buffer.append(user_turn)
            encoder_input.append(' '.join(encoder_buffer))
            encoder_buffer.append(system_turn)
            decoder_input.append(system_turn)
    return encoder_input, decoder_input


def get_files_list(in_root, in_task_id):
    result = []
    file_prefix = path.join(in_root, 'dialog-babi-{}'.format(in_task_id))
    for suffix in DATASET_ORDERING:
        task_file = '{}-{}.txt'.format(file_prefix, suffix)
        if os.path.exists(task_file):
            result.append(task_file)
    return result


def load_dataset(in_root, in_task_id):
    return list(map(read_task, get_files_list(in_root, in_task_id)))


# only returns filenames in a sorted order
def load_babble_dataset(in_root):
    files = os.listdir(in_root)
    data_parts = [
        sorted(
            filter(lambda x: '{}.txt'.format(suffix) in x, files),
            key=lambda x: int(x.split('.')[-1])
        )
        for suffix in DATASET_ORDERING
    ]
    return reduce(lambda x, y: x + y, data_parts, [])


def extract_slot_values(in_dialogues):
    result = defaultdict(lambda: set([]))
    for dialogue_name, dialogue in in_dialogues:
        for turn in dialogue:
            if not turn['text'].startswith('api_call'):
                continue
            slot_values = turn['text'].split()[1:]
            [result[index].add(value) for index, value in enumerate(slot_values)]
    return result


def save_csv(in_dialogue, out_stream):
    for turn in in_dialogue:
        print >>out_stream, '{}:\t{}'.format(turn['agent'], turn['text'])
