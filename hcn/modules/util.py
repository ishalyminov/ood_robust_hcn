import sys
from operator import itemgetter
import logging
import json
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.preprocessing import UtteranceEmbed, BoW_encoder, UNK_ID

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

EMBEDDING = None


def make_dataset_for_hcn(in_dialog_utterances, in_dialog_indices, in_vocab, in_entity_tracker, in_action_tracker, in_max_input_length):
    global EMBEDDING
    if not EMBEDDING:
        EMBEDDING = UtteranceEmbed()
    dialogs = [in_dialog_utterances[idx['start']: idx['end']] for idx in in_dialog_indices]

    bow_encoder = BoW_encoder(in_vocab)
    unk_action_id = in_action_tracker.action_templates.index('_UNK')

    X, prev_actions, action_masks, sequence_masks, y = [], [], [], [], []
    for dialog in dialogs:
        in_entity_tracker.reset()
        dialog_X, dialog_action_masks, dialog_y = [], [], []
        for utterance, action in dialog:
            utterance_delexicalized = in_entity_tracker.extract_entities(utterance)
            bow_features = bow_encoder.encode(utterance_delexicalized)
            embedding_features = EMBEDDING.encode(utterance_delexicalized)
            context_features = in_entity_tracker.context_features()
            action_mask = in_action_tracker.action_mask()
            turn_features = np.concatenate([bow_features, embedding_features, context_features], axis=-1)
            dialog_action_masks.append(action_mask)
            dialog_X.append(turn_features)
            action_delexicalized = in_entity_tracker.extract_entities(action, update=False)
            action_id = in_action_tracker.action_templates.index(action_delexicalized) \
                if action_delexicalized in in_action_tracker.action_templates \
                else unk_action_id

            dialog_y.append(action_id)
        dialog_prev_actions = tf.keras.utils.to_categorical([in_action_tracker.action_templates.index('_PAD')] + dialog_y[:-1],
                                                            num_classes=len(in_action_tracker.action_templates))
        X.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_X], padding='post', maxlen=in_max_input_length, dtype='float32')[0])
        prev_actions.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_prev_actions],
                                                                          padding='post',
                                                                          maxlen=in_max_input_length,
                                                                          dtype='float32')[0])
        action_masks.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_action_masks], padding='post', maxlen=in_max_input_length, dtype='float32')[0])
        y.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_y], padding='post', maxlen=in_max_input_length, dtype='int32')[0])
    return list(map(np.array, [X, prev_actions, action_masks, y]))


def make_dataset_for_hcn_v2(in_dialog_utterances, in_dialog_indices, in_vocab, in_entity_tracker, in_action_tracker, in_max_input_length):
    dialogs = [in_dialog_utterances[idx['start']: idx['end']] for idx in in_dialog_indices]

    input_words, context_features, action_masks, prev_action, y = [], [], [], [], []
    for dialog in dialogs:
        in_entity_tracker.reset()
        dialog_input_words, dialog_context_features, dialog_action_masks, dialog_prev_action, dialog_y = [], [], [], [], []
        for utterance, action in dialog:
            utterance_delexicalized = in_entity_tracker.extract_entities(utterance)
            utterance_vectorized = [in_vocab.get(word, UNK_ID) for word in utterance.delexicalized.split()]
            context_features = in_entity_tracker.context_features()
            action_mask = in_action_tracker.action_mask()

            dialog_action_masks.append(action_mask)
            dialog_input_words.append(turn_features)
            action_delexicalized = in_entity_tracker.extract_entities(action, update=False)
            if action_delexicalized in in_action_tracker.action_templates:
                action_id = in_action_tracker.action_templates.index(action_delexicalized)
            else:
                action_id = in_action_tracker.action_templates.index('_UNK')

            dialog_y.append(action_id)
        input_words.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_input_words], padding='post', maxlen=in_max_input_length, dtype='float32')[0])
        action_masks.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_action_masks], padding='post', maxlen=in_max_input_length, dtype='float32')[0])
        y.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_y], padding='post', maxlen=in_max_input_length, dtype='int32')[0])
    return np.array(X), np.array(action_masks), np.array(y)


def save_model(in_vocab, in_config, in_kb, in_action_templates, in_model_folder):
    if not os.path.exists(in_model_folder):
        os.makedirs(in_model_folder)
    with open(os.path.join(in_model_folder, 'config.json'), 'w', encoding='utf-8') as config_out:
        json.dump(in_config, config_out)
    with open(os.path.join(in_model_folder, 'kb.json'), 'w', encoding='utf-8') as kb_out:
        json.dump(in_kb, kb_out)
    with open(os.path.join(in_model_folder, 'actions.txt'), 'w', encoding='utf-8') as actions_out:
        print('\n'.join(in_action_templates), file=actions_out)
    with open(os.path.join(in_model_folder, 'vocab.txt'), 'w', encoding='utf-8') as vocab_out:
        print('\n'.join(in_vocab), file=vocab_out)


def load_model(in_model_folder):
    with open(os.path.join(in_model_folder, 'config.json'), encoding='utf-8') as config_in:
        config = json.load(config_in)
    with open(os.path.join(in_model_folder, 'kb.json'), encoding='utf-8') as kb_in:
        kb = json.load(kb_in)
    with open(os.path.join(in_model_folder, 'actions.txt'), encoding='utf-8') as actions_in:
        action_templates = [line.rstrip() for line in actions_in]
    with open(os.path.join(in_model_folder, 'vocab.txt'), encoding='utf-8') as vocab_in:
        vocab = [line.rstrip() for line in vocab_in]
    return vocab, kb, action_templates, config


def read_content():
    return ' '.join(get_utterances())


def read_dialogs(in_file, with_indices=False):
    def rm_index(row):
        return [' '.join(row[0].split(' ')[1:])] + row[1:]

    def filter_(rows):
        return list(filter(lambda row: 1 < len(row), rows))

    with open(in_file, encoding='utf-8') as dialogs_in:
        dialogs = dialogs_in.read().split('\n\n')

        line_counter = 0
        updated_dialogs = []
        dialog_indices = []
        for dialog in dialogs:
            dialog = dialog.rstrip()
            if not len(dialog):
                continue
            dialog_utterances = filter_([rm_index(row.lower().split('\t')) for row in dialog.split('\n')])
            updated_dialogs += dialog_utterances
            dialog_indices.append({'start': line_counter, 'end': line_counter + len(dialog_utterances)})
            line_counter += len(dialog_utterances)
        if with_indices:
            return updated_dialogs, dialog_indices
        return updated_dialogs


def get_utterances(dialogs):
    return list(map(itemgetter(0), dialogs))


def get_responses(dialogs):
    return list(map(itemgetter(1), dialogs))


def get_entities(in_kb_file):
    def filter_(items):
        return sorted(list(set([item for item in items if item and '_' not in item])))

    with open(in_kb_file) as f:
        return filter_([item.split('\t')[-1] for item in f.read().split('\n') ])
