import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

from .lstm_base import LSTMBase


class DialogLevelLSTM(LSTMBase):
    def __graph__(self):
        # tf.reset_default_graph()

        # entry points
        features_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.feature_vector_size], name='input_features')
        action_ = tf.placeholder(tf.int32, [None, self.max_input_length], name='ground_truth_action')
        prev_action_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='prev_action')
        action_mask_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='action_mask')

        # input projection
        Wi = tf.get_variable('Wi',
                             shape=[self.feature_vector_size + 2 * self.action_size, self.nb_hidden],
                             dtype=tf.float32,
                             initializer=xav())
        bi = tf.get_variable('bi',
                             shape=[self.nb_hidden],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.))

        all_inputs = tf.concat([features_, prev_action_, action_mask_], axis=-1)
        # add relu/tanh here if necessary
        projected_features = tf.tensordot(all_inputs, Wi, axes=1) + bi

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.nb_hidden, state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, projected_features, dtype=tf.float32)

        # output projection
        Wo = tf.get_variable('Wo',
                             shape=[self.nb_hidden, self.action_size],
                             dtype=tf.float32,
                             initializer=xav())
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

        # sequence_mask = tf.placeholder(tf.float32, [None, self.max_input_length], name='sequence_mask')
        action_seq_length = tf.count_nonzero(action_, -1)
        mask_fn = lambda l: tf.sequence_mask(l, self.max_input_length, dtype=tf.float32)
        sequence_mask = mask_fn(action_seq_length)
        # loss
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=action_, weights=sequence_mask, average_across_batch=False)

        # train op
        self.lr = tf.train.exponential_decay(self.config['learning_rate'],
                                             self.global_step,
                                             self.config.get('steps_before_decay', 0),
                                             self.config.get('learning_rate_decay', 1.0),
                                             staircase=True)
        optimizer = getattr(tf.train, self.config['optimizer'])(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.config['clip_norm'])
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # attach symbols to self
        self.loss = loss
        self.prediction = prediction
        self.probs = probs
        self.logits = logits
        self.sequence_mask_ = sequence_mask
        self.train_op = train_op

        # attach placeholders
        self.features_ = features_
        self.action_ = action_
        self.prev_action_ = prev_action_
        self.action_mask_ = action_mask_
    # forward propagation

    def forward(self, X, prev_action, action_mask, y):
        # forward
        probs, prediction, loss = self.sess.run([self.probs, self.prediction, self.loss],
                                                feed_dict={self.features_: X,
                                                           self.prev_action_: prev_action,
                                                           self.action_: y,
                                                           self.action_mask_: action_mask})
        return prediction, {'loss': loss}

    # training
    def train_step(self, X, prev_action, action_mask, y):
        _, loss_value, lr = self.sess.run([self.train_op, self.loss, self.lr],
                                          feed_dict={self.features_: X,
                                                     self.prev_action_: prev_action,
                                                     self.action_: y,
                                                     self.action_mask_: action_mask})
        # maintain state
        return {'loss': loss_value}, lr 

