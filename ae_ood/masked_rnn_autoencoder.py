import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append('..')


class MaskedRNNAutoencoder(object):
    """Basic version of LSTM-autoencoder.
      (cf. http://arxiv.org/abs/1502.04681)
    Usage:
      ae = LSTMAutoencoder(hidden_num, inputs)
      sess.run(ae.train)
    """
    MODEL_NAME = 'model.ckpt'
    VOCAB_NAME = 'vocab.txt'
    CONFIG_NAME = 'config.json'

    def __init__(self, config, vocab):
        self.build_graph(config, vocab)

    def build_graph(self, config, vocab):
        self.config = config
        self.vocab = vocab
        vocabulary_size = len(vocab)
        embedding_size = self.config['embedding_size']
        rnn_cell = self.config['rnn_cell']

        self._enc_cell = getattr(tf.contrib.rnn, rnn_cell)(embedding_size)
        self._dec_cell = getattr(tf.contrib.rnn, rnn_cell)(embedding_size)

        max_sequence_length = self.config['max_sequence_length']
        with tf.variable_scope('encoder'):
            self.X = tf.placeholder(tf.int32, [None, max_sequence_length], name='X')
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                                     name='emb')
            emb = tf.nn.embedding_lookup(embeddings, self.X)
            self.z_codes, self.enc_state = tf.nn.dynamic_rnn(self._enc_cell, emb, dtype=tf.float32)

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([embedding_size, vocabulary_size], dtype=tf.float32),
                                      name='dec_weight')
            dec_bias_ = tf.Variable(tf.random_uniform([vocabulary_size], -1.0, 1.0),
                                    name='dec_bias')

            dec_state = self.enc_state
            self.initial_dec_input = tf.placeholder(tf.float32, [None, embedding_size], name='dec_initial_input')
            dec_input_ = self.initial_dec_input
            dec_outputs = []
            for step in range(max_sequence_length):
                if step != 0:
                    vs.reuse_variables()
                dec_output, dec_state = self._dec_cell(dec_input_, dec_state)
                dec_output = tf.matmul(dec_output, dec_weight_) + dec_bias_
                dec_outputs.append(dec_output)
                dec_input_ = tf.nn.embedding_lookup(embeddings, tf.argmax(dec_output, axis=-1))
            self.output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

        self.enc_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.dec_seq_len = self.enc_seq_len + 1
        self.decoder_output_mask = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.dec_seq_len_weighted = tf.cast(self.dec_seq_len, dtype=tf.float64) * tf.cast(self.decoder_output_mask, dtype=tf.float64)
        mask_fn = lambda l: tf.sequence_mask(l, self.config['max_sequence_length'], dtype=tf.float32)
        mask = mask_fn(self.dec_seq_len)

        # loss for the whole batch
        self.loss_op = tf.contrib.seq2seq.sequence_loss(self.output_,
                                                        self.X,
                                                        weights=mask,
                                                        average_across_timesteps=True,
                                                        average_across_batch=False)
        self.opt = getattr(tf.train, self.config['optimizer'])(self.config['learning_rate'])
        self.train_op = self.opt.minimize(self.loss_op)

    def step(self, X, in_session, forward_only=False):
        if forward_only:
            batch_losses, output = in_session.run([self.loss_op, self.output_],
                                                  feed_dict={self.X: X,
                                                             self.initial_dec_input: np.zeros((X.shape[0], self.config['embedding_size']))})
            output_argmax = np.argmax(output, axis=-1)
            output_tokens = np.vectorize(self.vocab.__getitem__)(output_argmax)
            return batch_losses, output_tokens
        else:
            _, batch_losses = in_session.run([self.train_op, self.loss_op],
                                             feed_dict={self.X: X,
                                                        self.initial_dec_input: np.zeros((X.shape[0], self.config['embedding_size']))})
            return np.mean(batch_losses)

    def save(self, in_model_folder, in_session):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(in_session, os.path.join(in_model_folder, RNNAutoencoder.MODEL_NAME))
        with open(os.path.join(in_model_folder, RNNAutoencoder.VOCAB_NAME), 'w') as vocab_out:
            print('\n'.join(self.vocab), file=vocab_out)
        with open(os.path.join(in_model_folder, RNNAutoencoder.CONFIG_NAME), 'w') as config_out:
            json.dump(self.config, config_out)

    @staticmethod
    def load(in_model_folder, in_session):
        with open(os.path.join(in_model_folder, RNNAutoencoder.VOCAB_NAME), encoding='utf-8') as vocab_in:
            vocab = list(map(lambda x: x.strip(), vocab_in))
        with open(os.path.join(in_model_folder, RNNAutoencoder.CONFIG_NAME)) as config_in:
            config = json.load(config_in)
        ae = RNNAutoencoder(config, vocab)
        saver = tf.train.Saver()
        saver.restore(in_session, os.path.join(in_model_folder, RNNAutoencoder.MODEL_NAME))
        return ae

