import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.preprocessing import START_ID
import vae
from .lstm_base import LSTMBase


class VariationalHierarchicalLSTMv2(LSTMBase):
    def __init__(self, in_rev_vocab, in_config, in_feature_vector_size, in_action_size):
        self.rev_vocab = in_rev_vocab
        super(VariationalHierarchicalLSTMv2, self).__init__(in_config, in_feature_vector_size, in_action_size)

    def __graph__(self):
        with tf.variable_scope('vae'):
            self.vrae = getattr(vae, self.config['vae_model'])(self.config, self.rev_vocab, self.sess)
        # entry points
        input_words = tf.placeholder(tf.float32,
                                     [None, self.max_input_length, self.max_sequence_length],
                                     name='input_words')
        input_contexts = tf.placeholder(tf.float32,
                                        [None, self.max_input_length, self.feature_vector_size],
                                        name='input_contexts')
        action_ = tf.placeholder(tf.int32, [None, self.max_input_length], name='ground_truth_action')
        prev_action_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='prev_action')
        action_mask_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='action_mask')
        action_seq_length = tf.count_nonzero(action_, -1)

        vae_outputs = tf.reshape(self.vrae.z, shape=[-1, self.max_input_length, self.config['latent_size']]) 

        # input projection
        Wi_var = tf.get_variable('Wi_var',
                                 shape=[self.feature_vector_size + self.config['latent_size'] + 2 * self.action_size, self.config['dialog_level_embedding_size']],
                                 dtype=tf.float32,
                                 initializer=xav())
        bi_var = tf.get_variable('bi_var',
                                 shape=[self.config['dialog_level_embedding_size']],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.))

        turn_features_var = tf.concat([vae_outputs, input_contexts], axis=-1)
        all_inputs_var = tf.concat([turn_features_var, action_mask_, prev_action_], axis=-1)
        # add relu/tanh here if necessary
        projected_features_var = tf.tensordot(all_inputs_var, Wi_var, axes=1) + bi_var

        lstm_cell_var = tf.contrib.rnn.BasicLSTMCell(self.config['dialog_level_embedding_size'], state_is_tuple=True, name='dialog_encoder_var')
        outputs_var, states_var = tf.nn.dynamic_rnn(lstm_cell_var, projected_features_var, dtype=tf.float32)

        # output projection
        Wo = tf.get_variable('Wo',
                             shape=[self.config['dialog_level_embedding_size'], self.action_size],
                             dtype=tf.float32,
                             initializer=xav())
        bo = tf.get_variable('bo',
                             shape=[self.action_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.))
        logits = tf.tensordot(outputs_var, Wo, axes=1) + bo
        # probabilities
        #  normalization : elemwise multiply with action mask
        # not doing softmax because it's taken care of in the cross-entropy!
        probs = tf.multiply(logits, action_mask_)

        # prediction
        prediction = tf.argmax(probs, axis=-1)

        mask_fn = lambda l: tf.sequence_mask(l, self.max_input_length, dtype=tf.float32)
        sequence_mask = mask_fn(action_seq_length)
        # sequence_mask = tf.placeholder(tf.float32, [None, self.max_input_length], name='sequence_mask')
        # loss
        self.hcn_loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                         targets=action_,
                                                         weights=sequence_mask,
                                                         average_across_batch=False)
        self.vae_kl_loss = tf.reduce_mean(tf.reshape(self.vrae._kl_loss_fn(self.vrae.z_mean, self.vrae.z_logvar), shape=[-1, self.max_input_length]), axis=-1)
        # self.vae_nll_loss = tf.reduce_mean(tf.reshape(self.vrae._nll_loss_fn(), shape=[-1, self.max_input_length]), axis=-1)
        self.vae_bow_loss = tf.reduce_mean(tf.reshape(self.vrae._bow_loss_fn(self.vrae.bow_logits, self.vrae.bow_targets), shape=[-1, self.max_input_length]), axis=-1)
        self.vae_overall_loss = self.vae_kl_loss + self.vae_bow_loss # + self.vae_nll_loss
        self.loss = self.hcn_loss + self.vae_overall_loss
 
        self.lr = tf.train.exponential_decay(self.config['learning_rate'],
                                             self.global_step,
                                             self.config.get('steps_before_decay', 0),
                                             self.config.get('learning_rate_decay', 1.0),
                                             staircase=True) 
        # train op
        optimizer = getattr(tf.train, self.config['optimizer'])(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.config['clip_norm'])
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # attach symbols to self
        self.prediction = prediction
        self.probs = probs
        self.logits = logits
        self.sequence_mask_ = sequence_mask
        self.train_op = train_op

        # attach placeholders
        self.input_words = input_words
        self.input_contexts = input_contexts
        self.action_ = action_
        self.action_mask_ = action_mask_
        self.prev_action_ = prev_action_
    # forward propagation

    def forward(self, X, bow_out, context_features, action_mask, prev_action, y):
        enc_inp_flattened = X.reshape(X.shape[0] * X.shape[1], X.shape[-1])
        dec_inp_flattened = np.concatenate([np.expand_dims(np.array([START_ID] * enc_inp_flattened.shape[0]), -1), enc_inp_flattened[:,:-1]], axis=-1)
        bow_out_flattened = bow_out.reshape(bow_out.shape[0] * bow_out.shape[1], bow_out.shape[-1])
        dec_seq_length = np.ones(dec_inp_flattened.shape[0]) * dec_inp_flattened.shape[1]
        # forward
        probs, prediction, loss, hcn_loss, vae_kl_loss, vae_bow_loss = self.sess.run([self.probs, self.prediction, self.loss, self.hcn_loss, self.vae_kl_loss, self.vae_bow_loss],
                                                                                 feed_dict={self.input_words: X,
                                                                                            self.input_contexts: context_features,
                                                                                            self.vrae.enc_inp: enc_inp_flattened,
                                                                                            # self.vrae.dec_inp: dec_inp_flattened,
                                                                                            # self.vrae.dec_out: enc_inp_flattened,
                                                                                            self.vrae.bow_targets: bow_out_flattened,
                                                                                            # self.vrae.sequence_length_for_decoder: dec_seq_length,
                                                                                            self.action_: y,
                                                                                            self.prev_action_: prev_action,
                                                                                            self.action_mask_: action_mask})
        return prediction, {'loss': loss,
                            'hcn_loss': hcn_loss,
                            'vae_kl_loss': vae_kl_loss,
                            'vae_bow_loss': vae_bow_loss} 

    # training
    def train_step(self, X, bow_out, context_features, action_mask, prev_action, y):
        enc_inp_flattened = X.reshape(X.shape[0] * X.shape[1], X.shape[-1])
        dec_inp_flattened = np.concatenate([np.expand_dims(np.array([START_ID] * enc_inp_flattened.shape[0]), -1), enc_inp_flattened[:,:-1]], axis=-1)
        bow_out_flattened = bow_out.reshape(bow_out.shape[0] * bow_out.shape[1], bow_out.shape[-1])
        dec_seq_length = np.ones(dec_inp_flattened.shape[0]) * dec_inp_flattened.shape[1]
        _, loss_value, hcn_loss, vae_kl_loss, vae_bow_loss, lr = self.sess.run([self.train_op, self.loss, self.hcn_loss, self.vae_kl_loss, self.vae_bow_loss, self.lr],
                                                                           feed_dict={self.input_words: X,
                                                                                      self.input_contexts: context_features,
                                                                                      self.vrae.enc_inp: enc_inp_flattened,
                                                                                      # self.vrae.dec_inp: dec_inp_flattened,
                                                                                      # self.vrae.dec_out: enc_inp_flattened,
                                                                                      self.vrae.bow_targets: bow_out_flattened,
                                                                                      # self.vrae.sequence_length_for_decoder: dec_seq_length,
                                                                                      self.action_: y,
                                                                                      self.prev_action_: prev_action,
                                                                                      self.action_mask_: action_mask})
        # maintain state
        return {'loss': loss_value,
                'hcn_loss': hcn_loss,
                'vae_kl_loss': vae_kl_loss,
                'vae_bow_w': vae_bow_loss}, lr

