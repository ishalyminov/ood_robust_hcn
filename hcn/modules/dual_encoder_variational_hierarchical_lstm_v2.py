import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import vae
import ae_ood
from .lstm_base import LSTMBase


class DualEncoderVariationalHierarchicalLSTMv2(LSTMBase):
    def __init__(self, in_rev_vocab, in_config, in_feature_vector_size, in_action_size):
        self.rev_vocab = in_rev_vocab
        super(DualEncoderVariationalHierarchicalLSTMv2, self).__init__(in_config, in_feature_vector_size, in_action_size) 

    def __graph__(self):
        with tf.variable_scope('vae'):
            vrae = getattr(vae, self.config['vae_model'])(self.config, self.rev_vocab, self.sess)
        with tf.variable_scope('ae'):
            ae = ae_ood.RNNAutoencoder(self.config, self.rev_vocab)

        input_contexts = tf.placeholder(tf.float32,
                                        [None, self.max_input_length, self.feature_vector_size],
                                        name='input_contexts')
        action_ = tf.placeholder(tf.int32, [None, self.max_input_length], name='ground_truth_action')
        action_mask_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='action_mask')
        action_seq_length = tf.count_nonzero(action_, -1)

        vae_turn_encodings = vrae.z
        ae_turn_encodings = ae.enc_state.c
        turn_features = tf.concat([# tf.reshape(vae_turn_encodings, shape=[-1, self.max_input_length, self.config['latent_size']]),
                                   tf.reshape(ae_turn_encodings, shape=[-1, self.max_input_length, self.config['embedding_size']]),
                                   input_contexts],
                                  axis=-1)

        # input projection
        Wi = tf.get_variable('Wi',
                             shape=[self.feature_vector_size + self.nb_hidden, self.nb_hidden],
                             dtype=tf.float32,
                             initializer=xav())
        bi = tf.get_variable('bi',
                             shape=[self.nb_hidden],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.))

        # add relu/tanh here if necessary
        projected_features = tf.tensordot(turn_features, Wi, axes=1) + bi

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.nb_hidden, state_is_tuple=True, name='dialog_encoder')
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, projected_features, dtype=tf.float32)

        # output projection
        Wo = tf.get_variable('Wo',
                             shape=[self.nb_hidden, self.action_size],
                             dtype=tf.float32,
                             initializer=xav())
        # biased_action = np.zeros((self.action_size), dtype=np.float32)
        # biased_action[1] = 10.0
        bo = tf.get_variable('bo',
                             shape=[self.action_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.))
        # get logits
        logits = tf.tensordot(outputs, Wo, axes=1) + bo
        # probabilities
        #  normalization : elemwise multiply with action mask
        # not doing softmax because it's taken care of in the cross-entropy!
        probs = tf.multiply(logits, action_mask_)

        # prediction
        prediction = tf.argmax(probs, axis=-1)

        mask_fn = lambda l: tf.sequence_mask(l, self.max_input_length, dtype=tf.float32)
        sequence_mask = mask_fn(action_seq_length)
        # loss
        self.hcn_loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                         targets=action_,
                                                         weights=sequence_mask,
                                                         average_across_batch=False)

        self.vae_kl_loss = vrae._kl_loss_fn(vrae.z_mean, vrae.z_logvar)
        self.vae_bow_loss = vrae._bow_loss_fn(vrae.bow_logits, vrae.bow_targets)
        self.vae_overall_loss = self.vae_kl_loss + self.vae_bow_loss 
 
        # vae_loss = self.vae_nll_loss + self.vae_kl_w * self.vae_kl_loss
        loss = tf.reduce_mean(self.hcn_loss)  # + self.vae_overall_loss)
        self.lr = tf.train.exponential_decay(self.config['learning_rate'],
                                             self.global_step,
                                             self.config.get('steps_before_decay', 0),
                                             self.config.get('learning_rate_decay', 1.0),
                                             staircase=True)
        optimizer = getattr(tf.train, self.config['optimizer'])(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.config['clip_norm'])
        self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # attach symbols to self
        self.vrae = vrae
        self.ae = ae
        self.loss = loss
        self.prediction = prediction
        self.probs = probs
        self.logits = logits
        self.sequence_mask_ = sequence_mask

        # attach placeholders
        self.input_contexts = input_contexts
        self.action_ = action_
        self.action_mask_ = action_mask_

    # forward propagation
    def forward(self, enc_inp, dec_inp, dec_out, bow_out, context_features, action_mask, y):
        # forward
        enc_inp_flattened = enc_inp.reshape(enc_inp.shape[0] * enc_inp.shape[1], enc_inp.shape[-1])
        bow_out_flattened = bow_out.reshape(bow_out.shape[0] * bow_out.shape[1], bow_out.shape[-1])

        probs, prediction, loss, hcn_loss, vae_kl_loss, vae_bow_loss = self.sess.run([self.probs, self.prediction, self.loss, self.hcn_loss, self.vae_kl_loss, self.vae_bow_loss],
                                                                                                   feed_dict={self.vrae.enc_inp: enc_inp_flattened,
                                                                                                              self.vrae.bow_targets: bow_out_flattened,
                                                                                                              self.ae.X: enc_inp_flattened,
                                                                                                              self.input_contexts: context_features,
                                                                                                              self.action_: y,
                                                                                                              self.action_mask_: action_mask})
        return prediction, {'loss': np.ones_like(vae_kl_loss) * loss,
                            'hcn_loss': hcn_loss,
                            'vae_kl_loss': vae_kl_loss,
                            'vae_bow_loss': vae_bow_loss}

    # training
    def train_step(self, enc_inp, dec_inp, dec_out, bow_out, context_features, action_mask, y):
        enc_inp_flattened = enc_inp.reshape(enc_inp.shape[0] * enc_inp.shape[1], enc_inp.shape[-1])
        bow_out_flattened = bow_out.reshape(bow_out.shape[0] * bow_out.shape[1], bow_out.shape[-1])

        _, loss, hcn_loss, vae_kl_loss, vae_bow_loss, lr = self.sess.run([self.train_op, self.loss, self.hcn_loss, self.vae_kl_loss, self.vae_bow_loss, self.lr],
                                        feed_dict={self.vrae.enc_inp: enc_inp_flattened,
                                                   self.vrae.bow_targets: bow_out_flattened,
                                                   self.ae.X: enc_inp_flattened,
                                                   self.input_contexts: context_features,
                                                   self.action_: y,
                                                   self.action_mask_: action_mask})
        # maintain state
        return ({'loss': np.ones_like(vae_kl_loss) * loss,
                 'hcn_loss': hcn_loss,
                 'vae_kl_loss': vae_kl_loss,
                 'vae_bow_loss': vae_bow_loss},
                lr)

