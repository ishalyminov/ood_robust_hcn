import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav
from gensim.models import KeyedVectors

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import vae
import ae_ood
from .lstm_base import LSTMBase


class AE_HCN(LSTMBase):
    def __init__(self, in_rev_vocab, in_config, in_feature_vector_size, in_action_size):
        self.rev_vocab = in_rev_vocab
        super(AE_HCN, self).__init__(in_rev_vocab, in_config, in_feature_vector_size, in_action_size) 

    def __graph__(self):
        # entry points
        input_words = tf.placeholder(tf.int32,
                                     [None, self.max_input_length, self.max_sequence_length],
                                     name='input_words')
        # input_words_reshaped = tf.reshape(input_words, shape=[-1, self.max_sequence_length]) 
        input_contexts = tf.placeholder(tf.float32,
                                        [None, self.max_input_length, self.feature_vector_size],
                                        name='input_contexts')
        action_ = tf.placeholder(tf.int32, [None, self.max_input_length], name='ground_truth_action')
        action_mask_ = tf.placeholder(tf.float32, [None, self.max_input_length, self.action_size], name='action_mask')
        action_seq_length = tf.count_nonzero(action_, -1)

        ae_reconstruction_scores = tf.placeholder(tf.float32, [None, self.max_input_length, 1])
        turn_features = tf.concat([ae_reconstruction_scores, input_contexts], axis=-1)

        w2v_file = self.config.get('pretrained_embeddings')
        w2v_vectors = KeyedVectors.load_word2vec_format(w2v_file) if w2v_file else None
        self.initialize_from_w2v(w2v_vectors)

        # encoder
        self.embeddings = tf.Variable(self.embedding_matrix,
                                      dtype=tf.float32,
                                      name='emb')
        lookup_result = tf.nn.embedding_lookup(self.embeddings, input_words)
        turn_meanvector_encodings = tf.reduce_mean(lookup_result, axis=2)
        turn_features = tf.concat([turn_meanvector_encodings, ae_reconstruction_scores, input_contexts], axis=-1)
        # input projection
        Wi = tf.get_variable('Wi',
                             shape=[self.config['w2v_embedding_size'] + 1 + self.feature_vector_size, self.nb_hidden],
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
                                                         average_across_batch=True)
        loss = tf.reduce_mean(self.hcn_loss)
        # vae_loss = self.vae_nll_loss + self.vae_kl_w * self.vae_kl_loss
        self.lr = tf.train.exponential_decay(self.config['learning_rate'],
                                             self.global_step,
                                             self.config.get('steps_before_decay', 0),
                                             self.config.get('learning_rate_decay', 1.0),
                                             staircase=True)
        optimizer = getattr(tf.train, self.config['optimizer'])(self.lr)
        grads = optimizer.compute_gradients(loss)
        grads_clipped = [(tf.clip_by_norm(grad, self.config['clip_norm']), var)
                        for grad, var in grads]
        self.train_op = optimizer.apply_gradients(grads_clipped, global_step=self.global_step)

        # attach symbols to self
        self.loss = loss
        self.prediction = prediction
        self.probs = probs
        self.logits = logits
        self.sequence_mask_ = sequence_mask

        # attach placeholders
        self.input_words = input_words
        self.input_contexts = input_contexts
        self.action_ = action_
        self.action_mask_ = action_mask_
        self.ae_reconstruction_scores = ae_reconstruction_scores

    def initialize_from_w2v(self, in_w2v):
        # random at [-1.0, 1.0]
        emb_size = self.config['w2v_embedding_size']
        self.embedding_matrix = 2.0 * np.random.random((len(self.vocab), emb_size)).astype(np.float32) - 1.0

        if not in_w2v:
            return
        for word, idx in self.vocab.items():
            if word in in_w2v:
                self.embedding_matrix[idx] = in_w2v[word]

    # forward propagation
    def forward(self, tokens, ctx, action_masks, prev_actions, ae_scores, y, y_weight):
        # forward

        probs, prediction, loss = \
            self.sess.run([self.probs, self.prediction, self.loss],
                          feed_dict={self.input_words: tokens,
                                     self.input_contexts: ctx,
                                     self.ae_reconstruction_scores: np.expand_dims(ae_scores, -1),
                                     self.action_: y,
                                     self.action_mask_: action_masks})
        return prediction, {'loss': loss,
                            'hcn_loss': loss}

    # training
    def train_step(self, tokens, ctx, action_masks, prev_actions, ae_scores, y, y_weight):
        _, loss, lr = self.sess.run([self.train_op, self.loss, self.lr],
                                    feed_dict={self.input_words: tokens,
                                               self.input_contexts: ctx,
                                               self.ae_reconstruction_scores: np.expand_dims(ae_scores, -1),
                                               self.action_: y,
                                               self.action_mask_: action_masks})
        # maintain state
        return ({'loss': loss,
                 'hcn_loss': loss},
                lr)

