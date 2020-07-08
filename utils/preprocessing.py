import json
from collections import defaultdict, deque
import logging
from operator import itemgetter
import os

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PAD_ID = 0
START_ID = 1
UNK_ID = 2
EOS_ID = 3
PAD = '_PAD'
START = '_START'
UNK = '_UNK'
EOS = '_EOS'

EMBEDDING = None


class UtteranceEmbed(object):
    DEFAULT_W2V_FILE = os.path.join(os.path.dirname(__file__), '..', 'hcn', 'data', 'GoogleNews-vectors-negative300.bin')

    def __init__(self, in_vocab, fname=DEFAULT_W2V_FILE, dim=300):
        self.vocab = in_vocab
        self.dim = dim
        # load saved model
        print('Loading the word2vec model...')
        w2v_model = KeyedVectors.load_word2vec_format(fname, binary=True)
        self.initialize_embedding_matrix(w2v_model)

    def encode(self, utterance):
        embs = [self.embedding_matrix[self.vocab[word]] for word in utterance.split() if word and word in self.vocab]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

    def initialize_embedding_matrix(self, in_w2v_model):
        self.embedding_matrix = np.random.random((len(self.vocab), self.dim)).astype(np.float32)
        for word, idx in self.vocab.items():
            if word in in_w2v_model:
                self.embedding_matrix[idx] = in_w2v_model[word]

    def encode_utterance_seq(self, utterance):
        embs = [self.model[word] for word in utterance.split() if word and word in self.model]
        return embs


class BoW_encoder(object):
    def __init__(self, in_vocab):
        self.vocab = in_vocab
        self.vocab_size = len(self.vocab)

    def encode(self, utterance):
        bow = np.zeros([self.vocab_size], dtype=np.int32)
        for word in utterance.split(' '):
            if word in self.vocab:
                idx = self.vocab.get(word, UNK_ID)
                bow[idx] += 1
        return bow


def get_w2v_model(in_vocab):
    global EMBEDDING
    if not EMBEDDING:
        EMBEDDING = UtteranceEmbed(in_vocab)
    return EMBEDDING.embedding_matrix


def load_hcn_json(in_file):
    result = {'dialogs': []}
    with open(in_file) as json_in:
        data = json.load(json_in)
    for dialog in data['dialogs']:
        current_dialog = {'turns': []}
        if 'actions' not in result:
            result['actions'] = [PAD] + dialog['actions']
        for turn in dialog['turns']:
            if 'label' in turn:
                current_dialog['turns'].append({'label': turn['label'] + 1,
                                                'input': turn['input']['text'].lower(),
                                                'context_features': turn['context_features']})
        result['dialogs'].append(current_dialog)
    return result


def load_txt(in_filename):
    with open(in_filename, encoding='utf-8') as lines_in:
        lines = list(map(lambda x: x.lower().strip().split(), lines_in))
    return lines


def make_vocabulary(in_lines,
                    max_vocabulary_size,
                    special_tokens=(PAD, START, UNK, EOS),
                    frequency_threshold=0,
                    ngram_sizes=(1,)):
    freqdict = defaultdict(lambda: 0)

    for line in in_lines:
        ngram_windows = [deque([], maxlen=size) for size in ngram_sizes]
        for token in line:
            for window in ngram_windows:
                window.append(token)
                if len(window) == window.maxlen:
                    freqdict[' '.join(window)] += 1
    vocab = sorted(freqdict.items(), key=itemgetter(1), reverse=True)
    vocab = list(filter(lambda x: frequency_threshold < x[1], vocab))
    logger.info('{} tokens ({:.3f}% of the vocabulary) were filtered due to the frequency threshold'
                .format(len(freqdict) - len(vocab), 100.0 * (len(freqdict) - len(vocab)) / len(freqdict)))
    rev_vocab = (list(special_tokens) + list(filter(lambda x: x not in special_tokens, map(itemgetter(0), vocab))))[:max_vocabulary_size]
    vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    return vocab, rev_vocab


def vectorize_sequences(in_sequences, in_vocab, prepend=(), append=()):
    sequences_vectorized = []
    for sequence in in_sequences:
        sequences_vectorized.append(list(prepend) + [in_vocab.get(token, UNK_ID) for token in sequence] + list(append))
    return sequences_vectorized


def pad_sequences(in_sequences, in_max_input_length, value=PAD_ID):
    return tf.keras.preprocessing.sequence.pad_sequences(in_sequences,
                                                         value=value,
                                                         maxlen=in_max_input_length,
                                                         padding='post')


def pad_twice(in_dialog_feature, in_max_turn_length, in_max_dialog_length, result_shape=()):
    result = np.zeros(result_shape, dtype=np.int32)
    for dialog_idx, dialog in enumerate(in_dialog_feature):
        dialog_feature_padded = tf.keras.preprocessing.sequence.pad_sequences(dialog, padding='post', maxlen=in_max_turn_length)
        dialog_padded = tf.keras.preprocessing.sequence.pad_sequences([dialog_feature_padded], padding='post', maxlen=in_max_dialog_length)
        result[dialog_idx] = dialog_padded
    return result


def make_hcn_dataset(in_dialogs,
                     in_vocab,
                     in_ctx_features_vocab,
                     in_entity_tracker,
                     max_input_length,
                     max_sequence_length,
                     **config):
    ctx_bow_encoder = BoW_encoder(in_ctx_features_vocab)
    turn_bow_encoder = BoW_encoder(in_vocab)

    turn_tokens, turn_bow, ctx_features, action_masks, prev_actions, y = [], [], [], [], [], []
    for dialog_i in in_dialogs['dialogs']:
        turn_tokens_i, turn_bow_i, ctx_features_i, action_masks_i, actions_i, y_i = ([],
                                                                                     [],
                                                                                     [],
                                                                                     [],
                                                                                     [],
                                                                                     [])
        for turn in dialog_i['turns']:
            usr = turn['input']
            sys = turn['label']
            ctx = turn['context_features']

            utterance_delexicalized = in_entity_tracker.extract_entities(usr)
            utterance_vectorized = [in_vocab.get(token, UNK_ID) for token in
                                    utterance_delexicalized.split()]
            ctx_vectorized = ctx_bow_encoder.encode(' '.join(ctx))
            action_mask = [1] * len(in_dialogs['actions'])

            turn_tokens_i.append(utterance_vectorized)
            turn_bow_i.append(turn_bow_encoder.encode(utterance_delexicalized))
            ctx_features_i.append(ctx_vectorized)
            action_masks_i.append(action_mask)
            actions_i.append(sys)
            y_i.append(sys)

        prev_actions_i = [len(in_dialogs['actions']) - 1] + actions_i[:-1]
        prev_actions_i_vectorized = tf.keras.utils.to_categorical(prev_actions_i, num_classes=len(in_dialogs['actions']))
        turn_tokens.append(turn_tokens_i)
        turn_bow.append(turn_bow_i)
        ctx_features.append(ctx_features_i)
        action_masks.append(action_masks_i)
        prev_actions.append(prev_actions_i_vectorized)
        y.append(y_i)

    tokens_padded = pad_twice(turn_tokens,
                              max_sequence_length,
                              max_input_length,
                              (len(turn_tokens), max_input_length, max_sequence_length))
    bow_padded = tf.keras.preprocessing.sequence.pad_sequences(turn_bow,  maxlen=max_input_length, padding='post')
    ctx_padded = tf.keras.preprocessing.sequence.pad_sequences(ctx_features,  maxlen=max_input_length, padding='post')
    action_masks_padded = tf.keras.preprocessing.sequence.pad_sequences(action_masks,  maxlen=max_input_length, padding='post')
    prev_actions_padded = tf.keras.preprocessing.sequence.pad_sequences(prev_actions,  maxlen=max_input_length, padding='post')
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_input_length, padding='post')
    return [tokens_padded, bow_padded, ctx_padded, action_masks_padded, prev_actions_padded, y_padded]


def generate_dropout_turns(in_number, in_min_len, in_max_len, in_dialogs, in_vocab, in_word_dropout_prob):
    bow_encoder = BoW_encoder(in_vocab)

    total_utterances = []
    for dialog in in_dialogs['dialogs']:
        for turn in dialog['turns']:
            tokens = turn['input'].split()
            if in_min_len <= len(tokens) <= in_max_len:
                total_utterances.append(turn['input'])
    utterance_sequences, utterance_bows = [], []
    for utterance in np.random.choice(total_utterances, size=in_number, replace=True):
        utterance_dropped_out = [UNK
                                 if np.random.random() < in_word_dropout_prob else token
                                 for token in utterance.split()]
        token_ids_dropped_out = [in_vocab[token] for token in utterance_dropped_out]
        utterance_sequences.append(token_ids_dropped_out)
        utterance_bows.append(bow_encoder.encode(' '.join(utterance_dropped_out)))
    return [tf.keras.preprocessing.sequence.pad_sequences(utterance_sequences, maxlen=in_max_len, padding='post'),
            np.array(utterance_bows)]


def generate_word_scramble_turns(in_number, in_min_len, in_max_len, in_dialogs, in_vocab, in_word_dropout_prob):
    bow_encoder = BoW_encoder(in_vocab)
    vocab_list = list(in_vocab)

    utterance_lengths = []
    for dialog in in_dialogs['dialogs']:
        for turn in dialog['turns']:
            tokens = turn['input'].split()
            if in_min_len <= len(tokens) <= in_max_len:
                utterance_lengths.append(len(tokens))
    utterance_sequences, utterance_bows = [], []
    for utterance_length in np.random.choice(utterance_lengths, size=in_number, replace=True):
        utterance_dropped_out = [UNK
                                 if np.random.random() < in_word_dropout_prob
                                 else np.random.choice(vocab_list)
                                 for _ in range(utterance_length)]
        token_ids_dropped_out = [in_vocab[token] for token in utterance_dropped_out]
        utterance_sequences.append(token_ids_dropped_out)
        utterance_bows.append(bow_encoder.encode(' '.join(utterance_dropped_out)))
    return [tf.keras.preprocessing.sequence.pad_sequences(utterance_sequences, maxlen=in_max_len, padding='post'),
            np.array(utterance_bows)]


def make_autoencoder_dataset(in_dataset, in_vocab, max_sequence_length):
    encoder_input_vectorized = vectorize_sequences(in_dataset, in_vocab)
    decoder_input_vectorized = vectorize_sequences(in_dataset, in_vocab, prepend=[START_ID])
    decoder_output_vectorized = vectorize_sequences(in_dataset, in_vocab, append=[EOS_ID])
    encoder_input_padded = pad_sequences(encoder_input_vectorized, max_sequence_length)
    decoder_input_padded = pad_sequences(decoder_input_vectorized, max_sequence_length)
    decoder_output_padded = pad_sequences(decoder_output_vectorized, max_sequence_length)
    return encoder_input_padded, decoder_input_padded, decoder_output_padded
