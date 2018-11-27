import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

from .lstm_base import LSTMBase


class HierarchicalLSTM(LSTMBase):
    def __init__(self, in_config, in_feature_vector_size, in_action_size):
        super(HierarchicalLSTM, self).__init__(in_config, in_feature_vector_size, in_action_size)

    def __graph__(self):
        # tf.reset_default_graph()

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

        turn_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.nb_hidden, state_is_tuple=True, name='turn_encoder')
        turn_outputs, turn_states = tf.nn.dynamic_rnn(turn_lstm_cell, input_words, dtype=tf.float32)

        # input projection
        Wi = tf.get_variable('Wi',
                             shape=[self.feature_vector_size + self.nb_hidden + 2 * self.action_size, self.nb_hidden],
                             dtype=tf.float32,
                             initializer=xav())
        bi = tf.get_variable('bi',
                             shape=[self.nb_hidden],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.))

        turn_features = tf.concat([turn_outputs, input_contexts, action_mask_, prev_action_], axis=-1)

        # add relu/tanh here if necessary
        projected_features = tf.tensordot(turn_features, Wi, axes=1) + bi

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.nb_hidden, state_is_tuple=True, name='dialog_encoder')
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
        # loss
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=action_,
                                                weights=sequence_mask,
                                                average_across_batch=False)
        self.lr = tf.train.exponential_decay(self.config['learning_rate'],
                                             self.global_step,
                                             self.config.get('steps_before_decay', 0),
                                             self.config.get('learning_rate_decay', 1.0),
                                             staircase=True) 
        # train op
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
        self.input_words = input_words
        self.input_contexts = input_contexts
        self.action_ = action_
        self.prev_action_ = prev_action_
        self.action_mask_ = action_mask_
    # forward propagation

    def forward(self, X, context_features, action_mask, prev_action, y):
        # forward
        probs, prediction, loss = self.sess.run([self.probs, self.prediction, self.loss],
                                                feed_dict={self.input_words: X,
                                                           self.input_contexts: context_features,
                                                           self.action_: y,
                                                           self.prev_action_: prev_action,
                                                           self.action_mask_: action_mask})
        return prediction, {'loss': loss} 

    # training
    def train_step(self, X, context_features, action_mask, prev_action, y):
        _, loss_value, lr = self.sess.run([self.train_op, self.loss, self.lr],
                                          feed_dict={self.input_words: X,
                                                     self.input_contexts: context_features,
                                                     self.action_: y,
                                                     self.prev_action_: prev_action,
                                                     self.action_mask_: action_mask})
        # maintain state
        return {'loss': loss_value}, lr

