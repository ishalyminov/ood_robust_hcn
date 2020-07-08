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
    DEFAULT_W2V_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'GoogleNews-vectors-negative300.bin')

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


def load_txt(in_filename):
    with open(in_filename, encoding='utf-8') as lines_in:
        lines = list(map(lambda x: x.lower().strip().split(), lines_in))
    return lines


def read_dialogs(in_dataset):
    turns, indices = [], []
    for dialog in in_dataset['dialogs']:
        dialog_turns = []
        for turn, action in zip(dialog['turns'], dialog['actions_tokenized']):
            if 'input' not in turn:
                continue
            usr = turn['input']['text'].lower()
            sys = turn['label']
            ctx = turn['context_features']
            dialog_turns.append((usr, ctx, sys))
        indices.append({'start': len(turns), 'end': len(turns) + len(dialog_turns)})
        turns += dialog_turns
    return turns, indices


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


def make_dataset(in_dataset, in_vocab, max_sequence_length):
    sequences_vectorized = vectorize_sequences(in_dataset, in_vocab, append=[EOS_ID])
    sequences_padded = pad_sequences(sequences_vectorized, max_sequence_length)
    with tf.Session() as sess:
        sequence_masks = sess.run(tf.sequence_mask(lengths=list(map(len, sequences_vectorized)), maxlen=max_sequence_length))
    return sequences_padded, sequence_masks


def make_variational_autoencoder_dataset(in_dataset, in_vocab, max_sequence_length):
    encoder_input_vectorized = vectorize_sequences(in_dataset, in_vocab)
    decoder_input_vectorized = vectorize_sequences(in_dataset, in_vocab, prepend=[START_ID])
    decoder_output_vectorized = vectorize_sequences(in_dataset, in_vocab, append=[EOS_ID])
    bow_output = np.zeros((len(in_dataset), len(in_vocab)))
    for idx, utterance in enumerate(encoder_input_vectorized):
        for token in utterance:
            bow_output[idx][token] = 1.0
    encoder_input_padded = pad_sequences(encoder_input_vectorized, max_sequence_length)
    decoder_input_padded = pad_sequences(decoder_input_vectorized, max_sequence_length)
    decoder_output_padded = pad_sequences(decoder_output_vectorized, max_sequence_length)
    return encoder_input_padded, decoder_input_padded, decoder_output_padded, bow_output


def make_dataset_for_hierarchical_hcn(in_dialog_utterances,
                                      in_dialog_indices,
                                      in_vocab,
                                      in_entity_tracker,
                                      in_action_tracker,
                                      max_input_length,
                                      max_sequence_length,
                                      **kwargs):
    dialogs = [in_dialog_utterances[idx['start']: idx['end']] for idx in in_dialog_indices]

    X, context_features, action_masks, sequence_masks, prev_actions, y = [], [], [], [], [], []
    for dialog in dialogs:
        in_entity_tracker.reset()
        dialog_X, dialog_context_features, dialog_action_masks, dialog_y = [], [], [], []
        for utterance, action in dialog:
            utterance_delexicalized = in_entity_tracker.extract_entities(utterance)
            utterance_vectorized = [in_vocab.get(token, UNK_ID) for token in utterance_delexicalized.split()]
            dialog_context_features.append(in_entity_tracker.context_features())
            dialog_action_masks.append(in_action_tracker.action_mask())
            dialog_X.append(utterance_vectorized)
            action_delexicalized = in_entity_tracker.extract_entities(action, update=False)
            if action_delexicalized in in_action_tracker.action_templates:
                action_id = in_action_tracker.action_templates.index(action_delexicalized)
            else:
                action_id = in_action_tracker.action_templates.index('_UNK')
            dialog_y.append(action_id)
        dialog_prev_actions = tf.keras.utils.to_categorical([in_action_tracker.action_templates.index('_PAD')] + dialog_y[:-1],
                                                            num_classes=len(in_action_tracker.action_templates))
        prev_actions.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_prev_actions],
                                                                          padding='post',
                                                                          maxlen=max_input_length,
                                                                          dtype='float32')[0])
        turns_padded = tf.keras.preprocessing.sequence.pad_sequences(dialog_X, padding='post', maxlen=max_sequence_length, dtype='int32')
        dialog_padded = tf.keras.preprocessing.sequence.pad_sequences([turns_padded], padding='post', maxlen=max_input_length, dtype='int32')[0]
        X.append(dialog_padded)
        dialog_context_features_padded = tf.keras.preprocessing.sequence.pad_sequences([dialog_context_features], padding='post', maxlen=max_input_length, dtype='int32')[0]
        context_features.append(dialog_context_features_padded)
        action_masks.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_action_masks], padding='post', maxlen=max_input_length, dtype='float32')[0])
        y.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_y], padding='post', maxlen=max_input_length, dtype='int32')[0])
    return list(map(np.array, [X, context_features, action_masks, prev_actions, y]))


def make_dataset_for_de_vhcn_v3(in_dialog_utterances,
                                in_dialog_indices,
                                in_vocab,
                                in_entity_tracker,
                                in_action_tracker,
                                max_input_length,
                                max_sequence_length,
                                **kwargs):
    dialogs = [in_dialog_utterances[idx['start']: idx['end']] for idx in in_dialog_indices]

    X, bow_output, context_features, action_masks, sequence_masks, y = [], [], [], [], [], []
    for dialog in dialogs:
        in_entity_tracker.reset()
        dialog_X, dialog_context_features, dialog_action_masks, dialog_y = [], [], [], []
        for utterance, action in dialog:
            utterance_delexicalized = in_entity_tracker.extract_entities(utterance)
            utterance_vectorized = [in_vocab.get(token, UNK_ID) for token in utterance_delexicalized.split()]
            dialog_context_features.append(in_entity_tracker.context_features())
            dialog_action_masks.append(in_action_tracker.action_mask())
            dialog_X.append(utterance_vectorized)
            action_delexicalized = in_entity_tracker.extract_entities(action, update=False)
            if action_delexicalized in in_action_tracker.action_templates:
                action_id = in_action_tracker.action_templates.index(action_delexicalized)
            else:
                action_id = in_action_tracker.action_templates.index('_UNK')
            dialog_y.append(action_id)
        dialog_utterances_1hot = np.zeros((max_input_length, len(in_vocab)), dtype=np.float32)
        for idx, utterance in enumerate(dialog_X):
            for token in utterance:
                dialog_utterances_1hot[idx][token] = 1.0
        bow_output.append(dialog_utterances_1hot)
        turns_padded = tf.keras.preprocessing.sequence.pad_sequences(dialog_X, padding='post', maxlen=max_sequence_length, dtype='int32')
        dialog_padded = tf.keras.preprocessing.sequence.pad_sequences([turns_padded], padding='post', maxlen=max_input_length, dtype='int32')[0]
        X.append(dialog_padded)
        dialog_context_features_padded = tf.keras.preprocessing.sequence.pad_sequences([dialog_context_features], padding='post', maxlen=max_input_length, dtype='int32')[0]
        context_features.append(dialog_context_features_padded)
        action_masks.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_action_masks], padding='post', maxlen=max_input_length, dtype='float32')[0])
        y.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_y], padding='post', maxlen=max_input_length, dtype='int32')[0])
    return list(map(np.array, [X, bow_output, context_features, action_masks, y]))


def make_dataset_for_vhcn_v2(in_dialog_utterances,
                             in_dialog_indices,
                             in_vocab,
                             in_entity_tracker,
                             in_action_tracker,
                             max_input_length,
                             max_sequence_length,
                             **kwargs):
    dialogs = [in_dialog_utterances[idx['start']: idx['end']] for idx in in_dialog_indices]

    X, bow_output, context_features, action_masks, prev_actions, y = [], [], [], [], [], []
    for dialog in dialogs:
        in_entity_tracker.reset()
        dialog_X, dialog_context_features, dialog_action_masks, dialog_prev_actions, dialog_sequence_masks, dialog_y = [], [], [], [], [], []
        for utterance, action in dialog:
            utterance_delexicalized = in_entity_tracker.extract_entities(utterance)
            utterance_vectorized = [in_vocab.get(token, UNK_ID) for token in utterance_delexicalized.split()]
            dialog_context_features.append(in_entity_tracker.context_features())
            dialog_action_masks.append(in_action_tracker.action_mask())
            dialog_X.append(utterance_vectorized)
            #action_delexicalized = in_entity_tracker.extract_entities(action, update=False)
            #if action_delexicalized in in_action_tracker.action_templates:
            #    action_id = in_action_tracker.action_templates.index(action_delexicalized)
            #else:
            #    action_id = in_action_tracker.action_templates.index('_UNK')
            #dialog_y.append(action_id)
            dialog_y.append(action)
        dialog_utterances_1hot = np.zeros((max_input_length, len(in_vocab)), dtype=np.float32)
        dialog_prev_actions = tf.keras.utils.to_categorical([in_action_tracker.action_templates.index('_PAD')] + dialog_y[:-1],
                                                            num_classes=len(in_action_tracker.action_templates))
        prev_actions.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_prev_actions],
                                                                          padding='post',
                                                                          maxlen=max_input_length,
                                                                          dtype='float32')[0])
        for idx, utterance in enumerate(dialog_X):
            for token in utterance:
                dialog_utterances_1hot[idx][token] = 1.0
        bow_output.append(dialog_utterances_1hot)
        turns_padded = tf.keras.preprocessing.sequence.pad_sequences(dialog_X, padding='post', maxlen=max_sequence_length, dtype='int32')
        dialog_padded = tf.keras.preprocessing.sequence.pad_sequences([turns_padded], padding='post', maxlen=max_input_length, dtype='int32')[0]
        X.append(dialog_padded)
        dialog_context_features_padded = tf.keras.preprocessing.sequence.pad_sequences([dialog_context_features], padding='post', maxlen=max_input_length, dtype='int32')[0]
        context_features.append(dialog_context_features_padded)
        action_masks.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_action_masks], padding='post', maxlen=max_input_length, dtype='float32')[0])
        y.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_y], padding='post', maxlen=max_input_length, dtype='int32')[0])
    return list(map(np.array, [X, bow_output, context_features, action_masks, prev_actions, y]))


def make_dataset_for_variational_hcn(in_dialog_utterances,
                                     in_dialog_indices,
                                     in_vocab,
                                     in_entity_tracker,
                                     in_action_tracker,
                                     max_input_length,
                                     max_sequence_length,
                                     delexicalize_utterances=True,
                                     delexicalize_actions=True,
                                     **kwargs):
    dialogs = [in_dialog_utterances[idx['start']: idx['end']] for idx in in_dialog_indices]

    encoder_input, decoder_input, decoder_output, bow_output, context_features, action_masks, sequence_masks, y = ([],
                                                                                                                   [],
                                                                                                                   [],
                                                                                                                   [],
                                                                                                                   [],
                                                                                                                   [],
                                                                                                                   [],
                                                                                                                   [])
    for dialog in dialogs:
        in_entity_tracker.reset()
        dialog_utterances, dialog_context_features, dialog_action_masks, dialog_y = [], [], [], []
        for utterance, action in dialog:
            utterance_delexicalized = in_entity_tracker.extract_entities(utterance)
            dialog_context_features.append(in_entity_tracker.context_features())
            dialog_action_masks.append(in_action_tracker.action_mask())
            dialog_utterances.append(utterance_delexicalized if delexicalize_utterances else utterance)
            action_delexicalized = in_entity_tracker.extract_entities(action, update=False) if delexicalize_actions else action
            if action_delexicalized in in_action_tracker.action_templates:
                action_id = in_action_tracker.action_templates.index(action_delexicalized)
            else:
                action_id = in_action_tracker.action_templates.index('_UNK')
            dialog_y.append(action_id)

        encoder_input_vectorized = vectorize_sequences(dialog_utterances, in_vocab)
        decoder_input_vectorized = vectorize_sequences(dialog_utterances, in_vocab, prepend=[START_ID])
        decoder_output_vectorized = vectorize_sequences(dialog_utterances, in_vocab, append=[EOS_ID])

        dialog_utterances_1hot = np.zeros((max_input_length, len(in_vocab)), dtype=np.float32)
        for idx, utterance in enumerate(encoder_input_vectorized):
            for token in utterance:
                dialog_utterances_1hot[idx][token] = 1.0
        bow_output.append(dialog_utterances_1hot)

        encoder_input_padded = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_vectorized,
                                                                             padding='post',
                                                                             maxlen=max_sequence_length,
                                                                             dtype='int32')
        decoder_input_padded = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_vectorized,
                                                                             padding='post',
                                                                             maxlen=max_sequence_length,
                                                                             dtype='int32')
        decoder_output_padded = tf.keras.preprocessing.sequence.pad_sequences(decoder_output_vectorized,
                                                                              padding='post',
                                                                              maxlen=max_sequence_length,
                                                                              dtype='int32')

        dialog_enc_inp_padded = tf.keras.preprocessing.sequence.pad_sequences([encoder_input_padded],
                                                                              padding='post',
                                                                              maxlen=max_input_length,
                                                                              dtype='int32')[0]
        dialog_dec_inp_padded = tf.keras.preprocessing.sequence.pad_sequences([decoder_input_padded],
                                                                              padding='post',
                                                                              maxlen=max_input_length,
                                                                              dtype='int32')[0]
        dialog_dec_out_padded = tf.keras.preprocessing.sequence.pad_sequences([decoder_output_padded],
                                                                              padding='post',
                                                                              maxlen=max_input_length,
                                                                              dtype='int32')[0]
        encoder_input.append(dialog_enc_inp_padded)
        decoder_input.append(dialog_dec_inp_padded)
        decoder_output.append(dialog_dec_out_padded)

        dialog_context_features_padded = tf.keras.preprocessing.sequence.pad_sequences([dialog_context_features],
                                                                                       padding='post',
                                                                                       maxlen=max_input_length,
                                                                                       dtype='int32')[0]
        context_features.append(dialog_context_features_padded)
        action_masks.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_action_masks],
                                                                          padding='post',
                                                                          maxlen=max_input_length,
                                                                          dtype='float32')[0])
        y.append(tf.keras.preprocessing.sequence.pad_sequences([dialog_y],
                                                               padding='post',
                                                               maxlen=max_input_length,
                                                               dtype='int32')[0])
    return list(map(np.array, [encoder_input,
                               decoder_input,
                               decoder_output,
                               bow_output,
                               context_features,
                               action_masks,
                               y]))


def generate_random_input_for_variational_hcn(in_number, in_max_len, in_vocab, in_rev_vocab):
    utterances = []
    utterance_lengths = np.random.choice(range(in_max_len), size=in_number, replace=True)
    for length in utterance_lengths:
        utterances.append(np.random.choice(in_rev_vocab[EOS_ID:], size=length, replace=True))

    return make_variational_autoencoder_dataset(utterances, in_vocab, in_max_len)


def generate_dropout_turns_for_variational_hcn(in_number, in_max_len, in_utterances, in_vocab, in_word_dropout_prob):
    vocabulary_words_list = list(in_vocab.keys())
    utterances = []
    for utterance in np.random.choice(in_utterances, size=in_number, replace=True):
        utterance_dropped_out = [np.random.choice(vocabulary_words_list) if np.random.random() < in_word_dropout_prob else token
                                 for token in utterance.split()]
        utterances.append(utterance_dropped_out)

    return make_variational_autoencoder_dataset(utterances, in_vocab, in_max_len)


def generate_random_input_for_hierarchical_hcn(in_number, in_max_len, in_vocab, in_rev_vocab):
    utterances = []
    utterance_lengths = np.random.choice(range(in_max_len), size=in_number, replace=True)
    for length in utterance_lengths:
        utterances.append(np.random.choice(in_rev_vocab[EOS_ID + 1:], size=length, replace=True))

    return [make_variational_autoencoder_dataset(utterances, in_vocab, in_max_len)[0]]


def generate_dropout_turns_for_hierarchical_hcn(in_number, in_max_len, in_utterances, in_vocab, in_word_dropout_prob):
    vocabulary_words_list = list(in_vocab.keys())
    utterances = []
    for utterance in np.random.choice(in_utterances, size=in_number, replace=True):
        utterance_dropped_out = [np.random.choice(vocabulary_words_list) if np.random.random() < in_word_dropout_prob else token
                                 for token in utterance.split()]
        utterances.append(utterance_dropped_out)

    return [make_variational_autoencoder_dataset(utterances, in_vocab, in_max_len)[0]]    


def generate_dropout_turns_for_de_vhcn_v3(in_number, in_max_len, in_utterances, in_vocab, in_word_dropout_prob, min_length=3):
    vocabulary_words_list = list(in_vocab.keys())
    utterances = []
    utterances_for_sampling = [utterance for utterance in in_utterances if min_length <= len(utterance.split())]
    for utterance in np.random.choice(utterances_for_sampling, size=in_number, replace=True):
        utterance_dropped_out = [np.random.choice(vocabulary_words_list) if np.random.random() < in_word_dropout_prob else token
                                 for token in utterance.split()]
        utterances.append(utterance_dropped_out)

    vae_dataset = make_variational_autoencoder_dataset(utterances, in_vocab, in_max_len)
    return [vae_dataset[0], vae_dataset[-1]]


def generate_dropout_turns_for_vhcn_v3(in_number, in_max_len, in_utterances, in_vocab, in_word_dropout_prob):
    global EMBEDDING
    if not EMBEDDING:
        EMBEDDING = UtteranceEmbed()
    vocabulary_words_list = list(in_vocab.keys())
    utterances = []
    for utterance in np.random.choice(in_utterances, size=in_number, replace=True):
        utterance_dropped_out = [np.random.choice(vocabulary_words_list) if np.random.random() < in_word_dropout_prob else token
                                 for token in utterance.split()]
        utterances.append(utterance_dropped_out)

    embeddings = np.array([EMBEDDING.encode_utterance_seq(utt) for utterance in utterances])
    bags_of_words = make_variational_autoencoder_dataset(utterances, in_vocab, in_max_len)[-1]
    return [embeddings, bags_of_words]


def generate_random_input_for_hcn(in_number, in_max_len, in_vocab, in_rev_vocab):
    bow_encoder = BoW_encoder(in_vocab)
    embedding = UtteranceEmbed()

    utterances = []
    utterance_lengths = np.random.choice(range(in_max_len), size=in_number, replace=True)
    for length in utterance_lengths:
        utterance = ' '.join(np.random.choice(in_rev_vocab[EOS_ID + 1:], size=length, replace=True))
        bow_features = bow_encoder.encode(utterance)
        embedding_features = embedding.encode(utterance)
        utterances.append(np.concatenate([bow_features, embedding_features], axis=-1))
    return np.array(utterances)


def generate_dropout_turns_for_hcn(in_number, in_max_len, in_utterances, in_vocab, in_word_dropout_prob):
    bow_encoder = BoW_encoder(in_vocab)
    embedding = UtteranceEmbed()

    vocabulary_words_list = list(in_vocab.keys())
    utterances = []
    for utterance in np.random.choice(in_utterances, size=in_number, replace=True):
        utterance_dropped_out = ' '.join([np.random.choice(vocabulary_words_list) if np.random.random() < in_word_dropout_prob else token
                                          for token in utterance.split()])
        bow_features = bow_encoder.encode(utterance)
        embedding_features = embedding.encode(utterance)
        utterances.append(np.concatenate([bow_features, embedding_features], axis=-1))
    return np.array(utterances)
