import os
import sys

import tensorflow as tf

from .lstm_base import LSTMBase

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.preprocessing import get_w2v_model


class VariationalDialogLevelLSTM(LSTMBase):
    def __graph__(self):
        # entry points
        input_words_ = tf.placeholder(tf.int32, [None, self.max_input_length, self.max_sequence_length], name='input_words')
        bow_features_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.config['vocabulary_size']], name='bow_features')
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
        utterance_embeddings = tf.reduce_mean(lookup_result, axis=2)

        self.z_mean = tf.layers.dense(utterance_embeddings, self.config['latent_size'])
        self.z_logvar = tf.layers.dense(utterance_embeddings, self.config['latent_size'])
        gaussian_noise = tf.truncated_normal(tf.shape(self.z_logvar))
        self.z = self.z_mean + tf.exp(0.5 * self.z_logvar) * gaussian_noise

        bow_logits = tf.layers.dense(self.z, len(self.vocab), name='bow_logits')

        all_input = tf.concat([self.z, bow_features_, context_features_, prev_action_, action_mask_], axis=-1)
        # input projection
        projected_features = tf.layers.dense(all_input, self.nb_hidden, name='input_projection')

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.nb_hidden, state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, projected_features, dtype=tf.float32)

        # output projection
        logits = tf.layers.dense(outputs, self.action_size, name='logits')
        # probabilities
        #  normalization : elemwise multiply with action mask
        # not doing softmax because it's taken care of in the cross-entropy!
        probs = tf.multiply(logits, action_mask_)

        # prediction
        prediction = tf.argmax(probs, axis=-1)

        mask_fn = lambda l: tf.sequence_mask(l, self.max_input_length, dtype=tf.float32)
        # sequence_mask = mask_fn(action_seq_length)
        sequence_mask = tf.placeholder(tf.float32, [None, self.max_input_length], name='sequence_mask')
        # loss
        self.l2_loss = tf.reduce_sum([tf.nn.l2_loss(v)
                                      for v in tf.trainable_variables()
                                      if v.name[0] != 'b']) * self.config['l2_coef']
        self.kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar), axis=-1), axis=-1)
        self.bow_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=bow_features_, logits=bow_logits), axis=-1), axis=-1)
        self.hcn_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=action_, weights=sequence_mask, average_across_batch=False)
        loss = self.hcn_loss + self.l2_loss + self.kl_loss + self.bow_loss

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
        self.action_ = action_
        self.prev_action_ = prev_action_
        self.action_mask_ = action_mask_
        self.sequence_mask_ = sequence_mask
    # forward propagation

    def forward(self, X, bow_features, context_features, action_masks, prev_actions, y, y_weight):
        # forward
        probs, prediction, loss, hcn_loss, kl_loss, bow_loss = self.sess.run([self.probs, self.prediction, self.loss, self.hcn_loss, self.kl_loss, self.bow_loss],
                                                                             feed_dict={self.input_words_: X,
                                                                                        self.bow_features_: bow_features,
                                                                                        self.context_features_: context_features,
                                                                                        self.prev_action_: prev_actions,
                                                                                        self.action_: y,
                                                                                        self.action_mask_: action_masks,
                                                                                        self.sequence_mask_: y_weight})
        return prediction, {'loss': loss, 'hcn_loss': hcn_loss, 'kl_loss': kl_loss, 'bow_loss': bow_loss}

    # training
    def train_step(self, X, bow_features, context_features, action_masks, prev_actions, y, y_weight):
        _, loss_value, hcn_loss, kl_loss, bow_loss, lr = self.sess.run([self.train_op, self.loss, self.hcn_loss, self.kl_loss, self.bow_loss, self.lr],
                                                                       feed_dict={self.input_words_: X,
                                                                                  self.bow_features_: bow_features,
                                                                                  self.context_features_: context_features,
                                                                                  self.prev_action_: prev_actions,
                                                                                  self.action_: y,
                                                                                  self.action_mask_: action_masks,
                                                                                  self.sequence_mask_: y_weight})
        return {'loss': loss_value, 'hcn_loss': hcn_loss, 'kl_loss': kl_loss, 'bow_loss': bow_loss}, lr

