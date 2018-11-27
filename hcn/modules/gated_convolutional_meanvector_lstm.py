import os
import sys

import tensorflow as tf

from .lstm_base import LSTMBase

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.preprocessing import get_w2v_model
from .dialog_level_lstm import DialogLevelLSTM
from .convolutional_dialog_level_lstm import ConvolutionalDialogLevelLSTM


class GatedConvolutionalMeanvectorLSTM(LSTMBase):
    def __graph__(self):
        with tf.variable_scope('hcn'):
            self.hcn = DialogLevelLSTM(self.vocab, self.config, self.feature_vector_size, self.action_size, standalone=False)
        with tf.variable_scope('conv_hcn'):
            self.conv_hcn = ConvolutionalDialogLevelLSTM(self.vocab, self.config, self.feature_vector_size, self.action_size, standalone=False)
        self.alpha = tf.Variable(0.1, name='gate')
        logits = self.alpha * self.hcn.logits + (1.0 - self.alpha) * self.conv_hcn.logits

        # probabilities
        #  normalization : elemwise multiply with action mask
        # not doing softmax because it's taken care of in the cross-entropy!
        probs = tf.multiply(logits, self.hcn.action_mask_)

        # prediction
        prediction = tf.argmax(probs, axis=-1)

        self.l2_loss = tf.reduce_sum([tf.nn.l2_loss(v)
                                      for v in tf.trainable_variables()
                                      if v.name[0] != 'b']) * self.config['l2_coef']
        self.hcn_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=self.hcn.action_, weights=self.hcn.sequence_mask_, average_across_batch=False)
        loss = self.hcn_loss + self.l2_loss

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

    # forward propagation
    def forward(self, X, bow_features, context_features, action_masks, prev_actions, y, y_weight):
        # forward
        probs, prediction, loss = self.sess.run([self.probs, self.prediction, self.loss],
                                                feed_dict={self.hcn.input_words_: X,
                                                           self.hcn.bow_features_: bow_features,
                                                           self.hcn.context_features_: context_features,
                                                           self.hcn.prev_action_: prev_actions,
                                                           self.hcn.action_: y,
                                                           self.hcn.action_mask_: action_masks,
                                                           self.hcn.sequence_mask_: y_weight,
                                                           self.conv_hcn.input_words_: X,
                                                           self.conv_hcn.bow_features_: bow_features,
                                                           self.conv_hcn.context_features_: context_features,
                                                           self.conv_hcn.prev_action_: prev_actions,
                                                           self.conv_hcn.action_: y,
                                                           self.conv_hcn.action_mask_: action_masks,
                                                           self.conv_hcn.sequence_mask_: y_weight})
        return prediction, {'loss': loss}

    # training
    def train_step(self, X, bow_features, context_features, action_masks, prev_actions, y, y_weight):
        _, loss_value, hcn_loss, lr = self.sess.run([self.train_op, self.loss, self.hcn_loss, self.lr],
                                                    feed_dict={self.hcn.input_words_: X,
                                                               self.hcn.bow_features_: bow_features,
                                                               self.hcn.context_features_: context_features,
                                                               self.hcn.prev_action_: prev_actions,
                                                               self.hcn.action_: y,
                                                               self.hcn.action_mask_: action_masks,
                                                               self.hcn.sequence_mask_: y_weight,
                                                               self.conv_hcn.input_words_: X,
                                                               self.conv_hcn.bow_features_: bow_features,
                                                               self.conv_hcn.context_features_: context_features,
                                                               self.conv_hcn.prev_action_: prev_actions,
                                                               self.conv_hcn.action_: y,
                                                               self.conv_hcn.action_mask_: action_masks,
                                                               self.conv_hcn.sequence_mask_: y_weight,
                                                               self.conv_hcn.dropout_keep_prob: self.config['conv_dropout_keep_prob']})
        return {'loss': loss_value, 'hcn_loss': hcn_loss}, lr

