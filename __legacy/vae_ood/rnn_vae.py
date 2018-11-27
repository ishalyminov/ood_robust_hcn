import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append('..')

from utils.preprocessing import START_ID, EOS_ID


class RNNVAE(object):
    MODEL_NAME = 'model.ckpt'
    VOCAB_NAME = 'vocab.txt'
    CONFIG_NAME = 'config.json'

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        vocabulary_size = len(vocab)
        embedding_size = self.config['embedding_size']
        rnn_cell = self.config['rnn_cell']

        self._enc_cell = getattr(tf.contrib.rnn, rnn_cell)(embedding_size)
        self._dec_cell = getattr(tf.contrib.rnn, rnn_cell)(embedding_size)

        max_sequence_length = self.config['max_sequence_length']
        latent_size = self.config['latent_size']

        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope('encoder'):
            self.encoder_input = tf.placeholder(tf.int32, [None, max_sequence_length], name='encoder_input')
            self.decoder_input = tf.placeholder(tf.int32, [None, max_sequence_length], name='decoder_input')
            self.decoder_output = tf.placeholder(tf.int32, [None, max_sequence_length], name='decoder_output')
            self.enc_seq_len = tf.count_nonzero(self.encoder_input, 1, dtype=tf.int32)
            self.dec_seq_len = tf.count_nonzero(self.decoder_output, 1, dtype=tf.int32)
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                                     name='emb')
            _, self.enc_state = tf.nn.dynamic_rnn(self._enc_cell, tf.nn.embedding_lookup(embeddings, self.encoder_input), sequence_length=self.enc_seq_len, dtype=tf.float32)

            mean_dense = tf.layers.Dense(latent_size)
            logvar_dense = tf.layers.Dense(latent_size)
            enc_state_unified = tf.concat(self.enc_state, -1)
            self.z_mean =  mean_dense.apply(enc_state_unified)
            self.z_logvar = logvar_dense.apply(enc_state_unified)

            # reparametrization
            self.gaussian_noise = tf.truncated_normal(tf.shape(self.z_logvar))
            self.z = self.z_mean + tf.sqrt(tf.exp(self.z_logvar)) * self.gaussian_noise
        with tf.variable_scope('decoder'):
            init_state = tf.layers.dense(self.z, embedding_size * 2, tf.nn.elu)
            dec_state = tf.contrib.rnn.LSTMStateTuple(init_state[:, :embedding_size], init_state[:, embedding_size:])

            lin_proj = tf.layers.Dense(vocabulary_size)
            self.sequence_length_for_decoder = tf.placeholder(tf.int32, [None], name='decoder_sequence_length')
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=tf.nn.embedding_lookup(embeddings, self.decoder_input),
                                                       sequence_length=self.sequence_length_for_decoder)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=self._dec_cell,
                                                      helper=helper,
                                                      initial_state=dec_state,
                                                      output_layer=lin_proj)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
            self.output_ = decoder_output.rnn_output

            self.decoder_start_tokens = tf.placeholder(tf.int32, [None], name='decoder_start_token')
            decoder_forward_only = tf.contrib.seq2seq.BasicDecoder(cell=self._dec_cell,
                                                                   helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens=self.decoder_start_tokens, end_token=EOS_ID),
                                                                   initial_state=dec_state,
                                                                   output_layer=lin_proj)
            decoder_output_forward_only, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder_forward_only, maximum_iterations=max_sequence_length)
            self.output_forward_only_ = decoder_output_forward_only.rnn_output
 
        # self.loss_mask = tf.placeholder(tf.float32, [None, max_sequence_length], name='loss_mask')
        mask_fn = lambda l : tf.sequence_mask(l, max_sequence_length, dtype=tf.float32)
        mask = mask_fn(self.dec_seq_len)
        # loss for the whole batch
        self.nll_loss = tf.contrib.seq2seq.sequence_loss(self.output_,
                                                         self.decoder_output,
                                                         weights=mask,
                                                         average_across_timesteps=True,
                                                         average_across_batch=True,
                                                         softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
        anneal_rate, anneal_bias = self.config['anneal_rate'], self.config['anneal_bias']

        self.kl_loss = -0.5 * tf.reduce_mean(1 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar))
        self.kl_w = tf.sigmoid(anneal_rate * tf.to_float(self.global_step) + anneal_bias)
        self.loss_op = self.nll_loss  + self.kl_w * self.kl_loss
        self.opt = getattr(tf.train, self.config['optimizer'])(self.config['learning_rate'])
        self.train_op = self.opt.minimize(self.loss_op, global_step=self.global_step)


    def step(self, enc_input, dec_input, dec_output, in_session, forward_only=False):
        if forward_only:
            batch_losses, nll_losses, kl_losses, kl_w, output = in_session.run([self.loss_op, self.nll_loss, self.kl_loss, self.kl_w, self.output_forward_only_],
                                                                                feed_dict={self.encoder_input: enc_input,
                                                                                           self.decoder_input: dec_input,
                                                                                           self.decoder_output: dec_output,
                                                                                           self.sequence_length_for_decoder: np.ones(dec_output.shape[0]) * dec_output.shape[1],
                                                                                           self.decoder_start_tokens: np.ones(dec_input.shape[0]) * START_ID})
            output_argmax = np.argmax(output, axis=-1)
            output_tokens = np.vectorize(self.vocab.__getitem__)(output_argmax)
            return {'nll_loss': nll_losses, 'kl_loss': kl_losses, 'kl_w': kl_w}, output_tokens
        else:
            _, batch_losses, nll_losses, kl_losses, kl_w = in_session.run([self.train_op, self.loss_op, self.nll_loss, self.kl_loss, self.kl_w],
                                                                          feed_dict={self.encoder_input: enc_input, self.decoder_input: dec_input, self.decoder_output: dec_output,
                                                                                     self.sequence_length_for_decoder: np.ones(dec_output.shape[0]) * dec_output.shape[1]})
            return {'nll_loss': nll_losses, 'kl_loss': kl_losses, 'kl_w': kl_w}

    def save(self, in_model_folder, in_session):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(in_session, os.path.join(in_model_folder, RNNVAE.MODEL_NAME))
        with open(os.path.join(in_model_folder, RNNVAE.VOCAB_NAME), 'w') as vocab_out:
            print('\n'.join(self.vocab), file=vocab_out)
        with open(os.path.join(in_model_folder, RNNVAE.CONFIG_NAME), 'w') as config_out:
            json.dump(self.config, config_out)

    @staticmethod
    def load(in_model_folder, in_session):
        with open(os.path.join(in_model_folder, RNNVAE.VOCAB_NAME), encoding='utf-8') as vocab_in:
            vocab = list(map(lambda x: x.strip(), vocab_in))
        with open(os.path.join(in_model_folder, RNNVAE.CONFIG_NAME)) as config_in:
            config = json.load(config_in)
        ae = RNNVAE(config, vocab)
        saver = tf.train.Saver()
        saver.restore(in_session, os.path.join(in_model_folder, RNNVAE.MODEL_NAME))
        return ae

