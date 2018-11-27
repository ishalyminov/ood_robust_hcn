import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

from .lstm_base import LSTMBase

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.preprocessing import get_w2v_model


class DialogLevelLSTMv3(LSTMBase):
    def __init__(self, in_config, in_feature_vector_size, in_action_size, in_vocab, model_folder=None, trainable_vars=[]):
        self.vocab = in_vocab
        self.trainable_vars = trainable_vars
        super(DialogLevelLSTMv3, self).__init__(in_config, in_feature_vector_size, in_action_size)
        if model_folder:
            self.restore(model_folder)

    def __graph__(self):
        # entry points
        input_words_ = tf.placeholder(tf.int32, [None, self.max_input_length, self.max_sequence_length], name='input_words')
        bow_features_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.config['vocabulary_size']], name='bow_features')
        context_features_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.feature_vector_size], name='input_features')
        action_ = tf.placeholder(tf.int32, [None, self.max_input_length], name='ground_truth_action')
        prev_action_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='prev_action')
        action_mask_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='action_mask')
        action_seq_length = tf.count_nonzero(action_, -1)

        embedding_matrix = tf.get_variable('emb',
                                           initializer=tf.constant(get_w2v_model(self.vocab)),
                                           trainable=True)
        lookup_result = tf.nn.embedding_lookup(embedding_matrix, input_words_)
        masked_emb = tf.concat([tf.zeros([1, 1]), tf.ones([embedding_matrix.get_shape()[0] - 1, 1])], axis=0)
        mask_lookup_result = tf.nn.embedding_lookup(masked_emb, input_words_)
        lookup_result = tf.multiply(lookup_result, mask_lookup_result)
        utterance_embeddings = tf.reduce_mean(lookup_result, axis=2)

        all_input = tf.concat([utterance_embeddings, bow_features_, context_features_, prev_action_, action_mask_], axis=-1) 
        # input projection
        Wi = tf.get_variable('Wi',
                             shape=[self.feature_vector_size + self.config['w2v_embedding_size'] + self.config['vocabulary_size'] + 2 * self.action_size, self.nb_hidden],
                             dtype=tf.float32,
                             initializer=xav())
        bi = tf.get_variable('bi',
                             shape=[self.nb_hidden],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.))

        # add relu/tanh here if necessary
        projected_features = tf.tensordot(all_input, Wi, axes=1) + bi

        Wbow = tf.get_variable('Wbow',
                               shape=[self.nb_hidden, len(self.vocab)],
                               dtype=tf.float32,
                               initializer=xav())
        bbow = tf.get_variable('bbow',
                               shape=[len(self.vocab)],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.))
        bow_logits = tf.tensordot(projected_features, Wbow, axes=1) + bbow

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

        mask_fn = lambda l: tf.sequence_mask(l, self.max_input_length, dtype=tf.float32)
        sequence_mask = mask_fn(action_seq_length)
        # sequence_mask = tf.placeholder(tf.float32, [None, self.max_input_length], name='sequence_mask')
        # loss
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v)
                                 for v in tf.trainable_variables()
                                 if v.name[0] != 'b']) * self.config['l2_coef']
        hcn_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=action_, weights=sequence_mask, average_across_batch=False)
        bow_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=bow_logits, labels=bow_features_), axis=-1)
        
        loss = hcn_loss + l2_loss + bow_loss

        # train op
        self.lr = tf.train.exponential_decay(self.config['learning_rate'],
                                             self.global_step,
                                             self.config.get('steps_before_decay', 0),
                                             self.config.get('learning_rate_decay', 1.0),
                                             staircase=True)
        optimizer = getattr(tf.train, self.config['optimizer'])(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients_filtered, variables_filtered = [], []
        if len(self.trainable_vars):
            for gradient, variable in zip(gradients, variables):
                if variable.name in self.trainable_vars:
                    gradients_filtered.append(gradient)
                    variables_filtered.append(variable)
        else:
            gradients_filtered, variables_filtered = gradients, variables
        gradients, _ = tf.clip_by_global_norm(gradients_filtered, self.config['clip_norm'])
        train_op = optimizer.apply_gradients(zip(gradients_filtered, variables_filtered), global_step=self.global_step)

        # attach symbols to self
        self.loss = loss
        self.prediction = prediction
        self.probs = probs
        self.logits = logits
        self.train_op = train_op

        # attach placeholders
        self.input_words_ = input_words_
        self.context_features_ = context_features_
        self.bow_features_ = bow_features_
        self.action_ = action_
        self.prev_action_ = prev_action_
        self.action_mask_ = action_mask_
    # forward propagation

    def forward(self, X, bow_features, context_features, action_masks, prev_actions, y):
        # forward
        probs, prediction, loss = self.sess.run([self.probs, self.prediction, self.loss],
                                                feed_dict={self.input_words_: X,
                                                           self.bow_features_: bow_features,
                                                           self.context_features_: context_features,
                                                           self.prev_action_: prev_actions,
                                                           self.action_: y,
                                                           self.action_mask_: action_masks})
        return prediction, {'loss': loss}

    # training
    def train_step(self, X, bow_features, context_features, action_masks, prev_actions, y):
        _, loss_value, lr = self.sess.run([self.train_op, self.loss, self.lr],
                                          feed_dict={self.input_words_: X,
                                                     self.bow_features_: bow_features,
                                                     self.context_features_: context_features,
                                                     self.prev_action_: prev_actions,
                                                     self.action_: y,
                                                     self.action_mask_: action_masks})
        return {'loss': loss_value}, lr 

