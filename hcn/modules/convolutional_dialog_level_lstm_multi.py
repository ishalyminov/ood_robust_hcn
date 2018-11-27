import os
import sys

import numpy as np
import tensorflow as tf

from .lstm_base import LSTMBase

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.preprocessing import get_w2v_model


class MultitaskConvolutionalDialogLevelLSTM(LSTMBase):
    def __graph__(self):
        # entry points
        input_words_ = tf.placeholder(tf.int32, [None, self.max_input_length, self.max_sequence_length], name='input_words')
        bow_features_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.config['vocabulary_size']], name='bow_features')
        next_turn_bow_features_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.config['vocabulary_size']], name='next_turn_bow_features')
        context_features_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.feature_vector_size], name='input_features')
        action_ = tf.placeholder(tf.int32, [None, self.max_input_length], name='ground_truth_action')
        prev_action_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='prev_action')
        action_mask_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='action_mask')
        # action_seq_length = tf.count_nonzero(action_, -1)

        embedding_matrix = tf.get_variable('emb',
                                           initializer=tf.constant(get_w2v_model(self.vocab)),
                                           trainable=True)
        lookup_result = tf.nn.embedding_lookup(embedding_matrix, input_words_)
        masked_emb = tf.concat([tf.zeros([1, 1]), tf.ones([embedding_matrix.get_shape()[0] - 1, 1])], axis=0)
        mask_lookup_result = tf.nn.embedding_lookup(masked_emb, input_words_)
        lookup_result = tf.multiply(lookup_result, mask_lookup_result)

        filter_sizes = self.config['filter_sizes']
        num_filters = self.config['num_filters']
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        lookup_result_reshaped_expanded = tf.expand_dims(tf.reshape(lookup_result, shape=(-1, self.max_sequence_length, self.config['w2v_embedding_size'])), -1)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config['w2v_embedding_size'], 1, num_filters]
                W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
                b_conv = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b_conv")
                conv = tf.nn.conv2d(lookup_result_reshaped_expanded,
                                    W_conv,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b_conv), name="relu_conv")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.config['max_sequence_length'] - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
            h_drop_flat = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        h_drop = tf.reshape(h_drop_flat, shape=(-1, self.max_input_length, num_filters_total)) # h_drop = tf.reshape(h_drop_flat, shape=(-1, self.max_input_length, num_filters_total))

        all_input = tf.concat([h_drop, bow_features_, context_features_, prev_action_, action_mask_], axis=-1)
        # input projection
        projected_features = tf.layers.dense(all_input, self.nb_hidden, name='input_projection')

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.nb_hidden, state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, projected_features, dtype=tf.float32)

        # output projection
        logits = tf.layers.dense(outputs, self.action_size, name='logits')
        next_turn_bow_logits = tf.layers.dense(outputs, self.config['vocabulary_size'])
        # probabilities
        #  normalization : elemwise multiply with action mask
        # not doing softmax because it's taken care of in the cross-entropy!
        probs = tf.multiply(logits, action_mask_)

        # prediction
        prediction = tf.argmax(probs, axis=-1)

        # mask_fn = lambda l: tf.sequence_mask(l, self.max_input_length, dtype=tf.float32)
        # sequence_mask = mask_fn(action_seq_length)
        sequence_mask = tf.placeholder(tf.float32, [None, self.max_input_length], name='sequence_mask')
        # loss
        self.l2_loss = tf.reduce_sum([tf.nn.l2_loss(v)
                                      for v in tf.trainable_variables()
                                      if v.name[0] != 'b']) * self.config['l2_coef']
        # self.kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar), axis=-1), axis=-1)
        # self.bow_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=bow_features_, logits=bow_logits), axis=-1), axis=-1)
        self.hcn_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=action_, weights=sequence_mask, average_across_batch=False)
        next_turn_bow_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=next_turn_bow_features_, logits=next_turn_bow_logits), axis=-1), axis=-1)
        loss = self.hcn_loss + self.l2_loss + next_turn_bow_loss

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
        gradients_filtered, _ = tf.clip_by_global_norm(gradients_filtered, self.config['clip_norm'])
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
        self.next_turn_bow_features_ = next_turn_bow_features_
        self.action_ = action_
        self.prev_action_ = prev_action_
        self.action_mask_ = action_mask_
        self.sequence_mask_ = sequence_mask
    # forward propagation

    def forward(self, X, bow_features, context_features, action_masks, prev_actions, y, y_weight):
        dummy_bow_shape = list(bow_features.shape)
        dummy_bow_shape[-2] = 1
        next_turn_bow_features = np.concatenate([bow_features[:,1:,:], np.zeros(dummy_bow_shape)], axis=-2)
        # forward
        probs, prediction, loss, hcn_loss = self.sess.run([self.probs, self.prediction, self.loss, self.hcn_loss],
                                                                   feed_dict={self.input_words_: X,
                                                                              self.bow_features_: bow_features,
                                                                              self.next_turn_bow_features_: next_turn_bow_features,
                                                                              self.context_features_: context_features,
                                                                              self.prev_action_: prev_actions,
                                                                              self.action_: y,
                                                                              self.action_mask_: action_masks,
                                                                              self.sequence_mask_: y_weight})
        return prediction, {'loss': loss, 'hcn_loss': hcn_loss}

    # training
    def train_step(self, X, bow_features, context_features, action_masks, prev_actions, y, y_weight):
        dummy_bow_shape = list(bow_features.shape)
        dummy_bow_shape[-2] = 1
        next_turn_bow_features = np.concatenate([bow_features[:,1:,:], np.zeros(dummy_bow_shape)], axis=-2)
        _, loss_value, hcn_loss, lr = self.sess.run([self.train_op, self.loss, self.hcn_loss, self.lr],
                                                                       feed_dict={self.input_words_: X,
                                                                                  self.bow_features_: bow_features,
                                                                                  self.next_turn_bow_features_: next_turn_bow_features,
                                                                                  self.context_features_: context_features,
                                                                                  self.prev_action_: prev_actions,
                                                                                  self.action_: y,
                                                                                  self.action_mask_: action_masks,
                                                                                  self.sequence_mask_: y_weight,
                                                                                  self.dropout_keep_prob: self.config['conv_dropout_keep_prob']})
        return {'loss': loss_value, 'hcn_loss': hcn_loss}, lr

