import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append('..')

'''
  For using with VAE-like data
'''
class CompatibleRNNAutoencoder(object):
    """Basic version of LSTM-autoencoder.
      (cf. http://arxiv.org/abs/1502.04681)
    Usage:
      ae = LSTMAutoencoder(hidden_num, inputs)
      sess.run(ae.train)
    """
    MODEL_NAME = 'model.ckpt'
    VOCAB_NAME = 'vocab.txt'
    CONFIG_NAME = 'config.json'

    def __init__(self, config, vocab, w2v=None):
        self.config = config
        self.vocab = vocab
        self.initialize_from_w2v(w2v)
        self.build_graph(config, vocab)

    def build_graph(self, config, vocab):
        with tf.variable_scope('CompatibleRNNAutoencoder'):
            vocabulary_size = len(vocab)
            embedding_size = self.config['embedding_size']
            hidden_size = self.config['hidden_size']
            z_size = self.config['z_size']
            rnn_cell = self.config['rnn_cell']

            self._enc_cell = getattr(tf.contrib.rnn, rnn_cell)(hidden_size)
            self._dec_cell = getattr(tf.contrib.rnn, rnn_cell)(hidden_size)

            max_sequence_length = self.config['max_sequence_length']
            with tf.variable_scope('encoder'):
                self.enc_input = tf.placeholder(tf.int32, [None, max_sequence_length], name='enc_input')
                self.dec_output = tf.placeholder(tf.int32, [None, max_sequence_length], name='dec_output')
                self.embeddings = tf.placeholder(tf.float32, [vocabulary_size, embedding_size], name='emb')
                # tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='emb')
                emb = tf.nn.embedding_lookup(self.embeddings, self.enc_input)
                self.enc_outputs, self.enc_state = tf.nn.dynamic_rnn(self._enc_cell, emb, dtype=tf.float32)

                z_weight = tf.Variable(tf.truncated_normal([hidden_size, z_size], dtype=tf.float32),
                                       name='z_weight')
                z_bias = tf.Variable(tf.random_uniform([z_size], -1.0, 1.0),
                                     name='z_bias')
                self.z_codes = tf.matmul(self.enc_state, z_weight) + z_bias
            with tf.variable_scope('decoder') as vs:
                dec_z_weight = tf.Variable(tf.truncated_normal([z_size, hidden_size], dtype=tf.float32),
                                           name='dec_z_weight')
                dec_z_bias = tf.Variable(tf.random_uniform([hidden_size], -1.0, 1.0),
                                         name='dec_z_bias')
                dec_state = tf.matmul(self.z_codes, dec_z_weight) + dec_z_bias

                dec_weight_ = tf.Variable(tf.truncated_normal([hidden_size, vocabulary_size], dtype=tf.float32),
                                          name='dec_weight')
                dec_bias_ = tf.Variable(tf.random_uniform([vocabulary_size], -1.0, 1.0),
                                        name='dec_bias')

                self.initial_dec_input = tf.placeholder(tf.float32, [None, embedding_size], name='dec_initial_input')
                dec_input_ = self.initial_dec_input
                dec_outputs = []
                for step in range(max_sequence_length):
                    if step != 0:
                        vs.reuse_variables()
                    dec_output, dec_state = self._dec_cell(dec_input_, dec_state)
                    dec_output = tf.matmul(dec_output, dec_weight_) + dec_bias_
                    dec_outputs.append(dec_output)
                    dec_input_ = tf.nn.embedding_lookup(self.embeddings, tf.argmax(dec_output, axis=-1))
                self.output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

            self.dec_seq_len = tf.count_nonzero(self.dec_output, 1, dtype=tf.int32)
            mask_fn = lambda l: tf.sequence_mask(l, self.config['max_sequence_length'], dtype=tf.float32)
            mask = mask_fn(self.dec_seq_len)

            # loss for the whole batch
            self.loss_op = tf.contrib.seq2seq.sequence_loss(self.output_,
                                                            self.dec_output,
                                                            weights=mask,
                                                            average_across_timesteps=True,
                                                            average_across_batch=False)
            self.opt = getattr(tf.train, self.config['optimizer'])(self.config['learning_rate'])
        self.train_op = self.opt.minimize(self.loss_op)

    def step(self, enc_input, dec_output, in_session, forward_only=False):
        if forward_only:
            batch_losses, output = in_session.run([self.loss_op, self.output_],
                                                  feed_dict={self.enc_input: enc_input,
                                                             self.dec_output: dec_output,
                                                             self.embeddings: self.embedding_matrix, 
                                                             self.initial_dec_input: np.zeros((enc_input.shape[0], self.config['embedding_size']))})
            output_argmax = np.argmax(output, axis=-1)
            output_tokens = np.vectorize(self.vocab.__getitem__)(output_argmax)
            return batch_losses, output_tokens
        else:
            _, batch_losses = in_session.run([self.train_op, self.loss_op],
                                             feed_dict={self.enc_input: enc_input,
                                                        self.dec_output: dec_output,
                                                        self.embeddings: self.embedding_matrix,
                                                        self.initial_dec_input: np.zeros((enc_input.shape[0], self.config['embedding_size']))})
            return np.mean(batch_losses)

    def save(self, in_model_folder, in_session):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(in_session, os.path.join(in_model_folder, CompatibleRNNAutoencoder.MODEL_NAME))
        with open(os.path.join(in_model_folder, CompatibleRNNAutoencoder.VOCAB_NAME), 'w') as vocab_out:
            print('\n'.join(self.vocab), file=vocab_out)
        with open(os.path.join(in_model_folder, CompatibleRNNAutoencoder.CONFIG_NAME), 'w') as config_out:
            json.dump(self.config, config_out)

    def initialize_from_w2v(self, in_w2v):
        # random at [-1.0, 1.0]
        emb_size = self.config['embedding_size']
        self.embedding_matrix = 2.0 * np.random.random((len(self.vocab), emb_size)).astype(np.float32) - 1.0

        if not in_w2v:
            return
        for word, idx in self.vocab.items():
            if word in in_w2v:
                self.embedding_matrix[idx] = in_w2v_model[word]

    @staticmethod
    def load(in_model_folder, in_session):
        with open(os.path.join(in_model_folder, CompatibleRNNAutoencoder.VOCAB_NAME), encoding='utf-8') as vocab_in:
            vocab = list(map(lambda x: x.strip(), vocab_in))
        with open(os.path.join(in_model_folder, CompatibleRNNAutoencoder.CONFIG_NAME)) as config_in:
            config = json.load(config_in)
        ae = CompatibleRNNAutoencoder(config, vocab)
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CompatibleRNNAutoencoder'))
        saver.restore(in_session, os.path.join(in_model_folder, CompatibleRNNAutoencoder.MODEL_NAME))
        return ae

