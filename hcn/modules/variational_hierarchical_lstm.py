import sys
import os

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.preprocessing import get_w2v_model
from .lstm_base import LSTMBase


class VariationalHierarchicalLSTM(LSTMBase):
    def __graph__(self):
        # entry points
        input_words = tf.placeholder(tf.int32,
                                     [None, self.max_input_length, self.max_sequence_length],
                                     name='input_words')
        input_contexts = tf.placeholder(tf.float32,
                                        [None, self.max_input_length, self.feature_vector_size],
                                        name='input_contexts')
        bow_features = tf.placeholder(tf.float32,
                                      [None, self.max_input_length, len(self.vocab)],
                                      name='bow_features')
        action_ = tf.placeholder(tf.int32, [None, self.max_input_length], name='ground_truth_action')
        prev_action_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='prev_action')
        action_mask_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='action_mask')
        # action_seq_length = tf.count_nonzero(action_, -1)

        input_words_reshaped = tf.reshape(input_words, shape=[-1, self.max_sequence_length])
        embedding_matrix = tf.get_variable('emb',
                                           initializer=tf.constant(get_w2v_model(self.vocab)),
                                           trainable=self.config['trainable_embeddings'])
        lookup_result = tf.nn.embedding_lookup(embedding_matrix, input_words_reshaped)
        masked_emb = tf.concat([tf.zeros([1, 1]), tf.ones([embedding_matrix.get_shape()[0] - 1, 1])], axis=0)
        mask_lookup_result = tf.nn.embedding_lookup(masked_emb, input_words_reshaped)
        lookup_result = tf.multiply(lookup_result, mask_lookup_result)

        self.turn_level_proj = tf.layers.dense(lookup_result, self.nb_hidden, name='turn_level_proj')

        if self.config['bi_lstm']:
            self.turn_fw_cell = tf.contrib.rnn.BasicLSTMCell(int(self.nb_hidden / 2),
                                                             state_is_tuple=True,
                                                             name='turn_encoder_fw')
            self.turn_bw_cell = tf.contrib.rnn.BasicLSTMCell(int(self.nb_hidden / 2),
                                                             state_is_tuple=True,
                                                             name='turn_encoder_bw')
            _, turn_states = tf.nn.bidirectional_dynamic_rnn(self.turn_fw_cell,
                                                             self.turn_bw_cell,
                                                             self.turn_level_proj,
                                                             dtype=tf.float32)
            fw_states, bw_states = turn_states
            fw_states_reshaped = tf.reshape(fw_states.c, shape=[-1,
                                                                self.max_input_length,
                                                                int(self.nb_hidden / 2)])
            bw_states_reshaped = tf.reshape(bw_states.c, shape=[-1,
                                                                self.max_input_length,
                                                                int(self.nb_hidden / 2)])
            self.turn_states_reshaped = tf.concat([fw_states_reshaped, bw_states_reshaped], axis=-1)
        else:
            self.turn_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.nb_hidden, state_is_tuple=True,
                                                               name='turn_encoder')
            turn_outputs, turn_states = tf.nn.dynamic_rnn(self.turn_lstm_cell,
                                                          self.turn_level_proj,
                                                          dtype=tf.float32)
            self.turn_states_reshaped = tf.reshape(turn_states.c, shape=[-1,
                                                                         self.max_input_length,
                                                                         self.nb_hidden])

        self.z_mean = tf.layers.dense(self.turn_states_reshaped,
                                      self.config['latent_size'],
                                      name='z_mean')
        self.z_logvar = tf.layers.dense(self.turn_states_reshaped,
                                        self.config['latent_size'],
                                        name='z_logvar')
        gaussian_noise = tf.truncated_normal(tf.shape(self.z_logvar))
        self.z = self.z_mean + tf.exp(0.5 * self.z_logvar) * gaussian_noise

        bow_logits = tf.layers.dense(self.z, len(self.vocab), name='bow_logits')

        # input projection
        all_inputs = tf.concat([self.z, bow_features, input_contexts, action_mask_, prev_action_], axis=-1)
        # add relu/tanh here if necessary
        dialog_lstm_projection = tf.layers.dense(all_inputs, self.config['embedding_size'], name='dialog_lstm_projection')

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config['embedding_size'], state_is_tuple=True, name='dialog_encoder')
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, dialog_lstm_projection, dtype=tf.float32)

        # output projection
        logits = tf.layers.dense(outputs, self.action_size, name='output_projection') 
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
        self.hcn_loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                         targets=action_,
                                                         weights=sequence_mask,
                                                         average_across_batch=False)
        # self.vae_kl_loss = tf.reduce_mean(tf.reshape(self.vrae._kl_loss_fn(self.vrae.z_mean, self.vrae.z_logvar), shape=[-1, self.max_input_length]), axis=-1)
        # self.vae_bow_loss = tf.reduce_mean(tf.reshape(self.vrae._bow_loss_fn(self.vrae.bow_logits, self.vrae.bow_targets), shape=[-1, self.max_input_length]), axis=-1)
        self.vae_kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + self.z_logvar - tf.square(self.z_mean) - tf.exp(self.z_logvar), axis=-1), axis=-1)
        self.vae_bow_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=bow_features, logits=bow_logits), axis=-1), axis=-1)
        self.vae_overall_loss = self.vae_kl_loss + self.vae_bow_loss

        self.l2_loss = tf.reduce_sum([tf.nn.l2_loss(v)
                                      for v in tf.trainable_variables()
                                      if v.name[0] != 'b']) * self.config['l2_coef']
        self.loss = self.hcn_loss + self.vae_overall_loss + self.l2_loss
 
        self.lr = tf.train.exponential_decay(self.config['learning_rate'],
                                             self.global_step,
                                             self.config.get('steps_before_decay', 0),
                                             self.config.get('learning_rate_decay', 1.0),
                                             staircase=True) 
        # train op
        optimizer = getattr(tf.train, self.config['optimizer'])(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
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
        self.prediction = prediction
        self.probs = probs
        self.logits = logits
        self.sequence_mask_ = sequence_mask
        self.train_op = train_op

        # attach placeholders
        self.input_words = input_words
        self.input_contexts = input_contexts
        self.bow_features_ = bow_features
        self.action_ = action_
        self.action_mask_ = action_mask_
        self.prev_action_ = prev_action_
    # forward propagation

    def forward(self, X, bow_out, context_features, action_mask, prev_action, y, y_weight):
        # dec_seq_length = np.ones(dec_inp_flattened.shape[0]) * dec_inp_flattened.shape[1]
        # forward
        result = self.sess.run([self.probs, self.prediction, self.loss, self.hcn_loss, self.vae_kl_loss, self.vae_bow_loss],
                               feed_dict={self.input_words: X,
                                          self.input_contexts: context_features,
                                          self.action_: y,
                                          self.bow_features_: bow_out,
                                          self.prev_action_: prev_action,
                                          self.action_mask_: action_mask,
                                          self.sequence_mask_: y_weight})
        probs, prediction, loss, hcn_loss, vae_kl_loss, vae_bow_loss = result
        return prediction, {'loss': loss,
                            'hcn_loss': hcn_loss,
                            'vae_kl_loss': vae_kl_loss,
                            'vae_bow_loss': vae_bow_loss} 

    # training
    def train_step(self, X, bow_out, context_features, action_mask, prev_action, y, y_weight):
        # dec_seq_length = np.ones(dec_inp_flattened.shape[0]) * dec_inp_flattened.shape[1]
        result = self.sess.run([self.train_op, self.loss, self.hcn_loss, self.vae_kl_loss, self.vae_bow_loss, self.lr],
                               feed_dict={self.input_words: X,
                                          self.input_contexts: context_features,
                                          self.action_: y,
                                          self.bow_features_: bow_out,
                                          self.prev_action_: prev_action,
                                          self.action_mask_: action_mask,
                                          self.sequence_mask_: y_weight})
        _, loss_value, hcn_loss, vae_kl_loss, vae_bow_loss, lr = result
        return {'loss': loss_value,
                'hcn_loss': hcn_loss,
                'vae_kl_loss': vae_kl_loss,
                'vae_bow_loss': vae_bow_loss}, lr

