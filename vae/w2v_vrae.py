import json
import os

from vae.modified import ModifiedBasicDecoder, ModifiedBeamSearchDecoder

import tensorflow as tf
import numpy as np

from utils.preprocessing import PAD_ID, START_ID, EOS_ID, UNK_ID, get_w2v_model


class W2VRAE(object):
    MODEL_NAME = 'vrae.ckpt'
    VOCAB_NAME = 'vocab.txt'
    CONFIG_NAME = 'config.json'

    def __init__(self, config, rev_vocab, session, standalone=False):
        self.vocab = {word: idx for idx, word in enumerate(rev_vocab)}
        self.rev_vocab = rev_vocab
        self.config = config
        self.session = session
        self._build_graph()
        if standalone:
            session.run(tf.global_variables_initializer())

    def _build_graph(self):
        with tf.variable_scope('W2VRAE', reuse=tf.AUTO_REUSE):
            self._build_forward_graph()
        self._build_backward_graph()

    def _build_forward_graph(self):
        self._build_inputs()
        self._decode(self._reparam(*self._encode()))

    def _build_backward_graph(self):
        with tf.variable_scope('W2VRAE', reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, trainable=False)
            self.nll_loss = self._nll_loss_fn()
            self.kl_w = self._kl_w_fn(self.config['anneal_max'], self.config['anneal_bias'], self.global_step)
            self.kl_loss = self._kl_loss_fn(self.z_mean, self.z_logvar)
            self.bow_loss = self._bow_loss_fn(self.bow_logits, self.bow_targets)
            self.overall_loss = self.nll_loss + self.kl_loss + self.bow_loss
            self.batch_loss = tf.reduce_mean(self.overall_loss)
            clipped_gradients, params = self._gradient_clipping(self.batch_loss)
            optimizer = getattr(tf.train, self.config['optimizer'])(self.config['learning_rate'])
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def _build_inputs(self):
        # placeholders
        self.enc_inp = tf.placeholder(tf.int32, [None, self.config['max_sequence_length']])
        self.dec_inp = tf.placeholder(tf.int32, [None, self.config['max_sequence_length']])
        self.dec_out = tf.placeholder(tf.int32, [None, self.config['max_sequence_length']])
        self.bow_targets = tf.placeholder(tf.float32, [None, self.config['vocabulary_size']])
        # global helpers
        self._batch_size = tf.shape(self.enc_inp)[0]
        self.enc_seq_len = tf.count_nonzero(self.enc_inp, 1, dtype=tf.int32)
        self.dec_seq_len = self.enc_seq_len + 1

    def _encode(self):
        # the embedding is shared between encoder and decoder
        # since the source and the target for an autoencoder are the same
        with tf.variable_scope('encoder'):
            tied_embedding = tf.get_variable('tied_embedding',
                                             initializer=tf.constant(get_w2v_model(self.vocab)),
                                             trainable=True)
            lookup_result = tf.nn.embedding_lookup(tied_embedding, self.enc_inp)
            masked_emb = tf.concat([tf.zeros([1, 1]), tf.ones([tied_embedding.get_shape()[0] - 1, 1])], axis=0)
            mask_lookup_result = tf.nn.embedding_lookup(masked_emb, self.enc_inp)
            lookup_result = tf.multiply(lookup_result, mask_lookup_result)
            encoder_proj = tf.layers.dense(lookup_result, self.config['rnn_size'])

            if self.config.get('bi_lstm'):
                forward_encoder = self._rnn_cell(self.config['w2v_embedding_size'] // 2)
                backward_encoder = self._rnn_cell(self.config['w2v_embedding_size'] // 2)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_encoder,
                                                                          cell_bw=backward_encoder,
                                                                          inputs=encoder_proj,
                                                                          sequence_length=self.enc_seq_len,
                                                                          dtype=tf.float32)
                encoded_state = tf.concat((state_fw, state_bw), -1)
            else:
                _, encoded_state = tf.nn.dynamic_rnn(cell=self._rnn_cell(self.config['w2v_embedding_size']),
                                                     inputs=encoder_proj,
                                                     sequence_length=self.enc_seq_len,
                                                     dtype=tf.float32)
        self.lookup_result = lookup_result
        self.z_mean = tf.layers.dense(encoded_state, self.config['latent_size'])
        self.z_logvar = tf.layers.dense(encoded_state, self.config['latent_size'])
        return self.z_mean, self.z_logvar

    def _reparam(self, z_mean, z_logvar):
        self.gaussian_noise = tf.truncated_normal(tf.shape(z_logvar))
        self.z = z_mean + tf.exp(0.5 * z_logvar) * self.gaussian_noise
        return self.z

    def _decode(self, z):
        self.training_rnn_out, self.training_logits = self._decoder_training(z)
        self.predicted_ids = self._decoder_inference(z)

    def _decoder_training(self, z):
        with tf.variable_scope('encoder', reuse=True):
            tied_embedding = tf.get_variable('tied_embedding', [self.config['vocabulary_size'], self.config['w2v_embedding_size']])

        with tf.variable_scope('decoding'):
            self.decoder_init_state = tf.layers.dense(self.z, self.config['w2v_embedding_size'], tf.nn.elu)

            lin_proj = tf.layers.Dense(self.config['vocabulary_size'], _scope='decoder/dense')
            self.bow_logits = tf.layers.dense(self.z, self.config['vocabulary_size'], name='bow_logits')
            self.sequence_length_for_decoder = tf.placeholder(tf.int32, [None], name='decoder_sequence_length')
            decoder_emb = tf.nn.embedding_lookup(tied_embedding, self.dec_inp)
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_emb,
                                                       sequence_length=self.sequence_length_for_decoder)
            decoder = ModifiedBasicDecoder(cell=self._rnn_cell(),
                                           helper=helper,
                                           initial_state=self.decoder_init_state,
                                           concat_z=self.z)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
        return decoder_output.rnn_output, lin_proj.apply(decoder_output.rnn_output)

    def _decoder_inference(self, z):
        tiled_z = tf.tile(tf.expand_dims(self.z, 1), [1, self.config['beam_width'], 1])

        with tf.variable_scope('encoder', reuse=True):
            tied_embedding = tf.get_variable('tied_embedding',
                                             [self.config['vocabulary_size'], self.config['w2v_embedding_size']])

        with tf.variable_scope('decoding', reuse=True):
            init_state = tf.layers.dense(self.z, self.config['w2v_embedding_size'], tf.nn.elu, reuse=True)

            decoder = ModifiedBeamSearchDecoder(cell=self._rnn_cell(reuse=True),
                                                embedding=tied_embedding,
                                                start_tokens=tf.tile(tf.constant([START_ID], dtype=tf.int32),
                                                                     [self._batch_size]),
                                                end_token=EOS_ID,
                                                initial_state=tf.contrib.seq2seq.tile_batch(init_state, self.config['beam_width']),
                                                beam_width=self.config['beam_width'],
                                                output_layer=tf.layers.Dense(self.config['vocabulary_size'], _reuse=True),
                                                concat_z=tiled_z)
            max_sequence_length = self.config['max_sequence_length']
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                     maximum_iterations=max_sequence_length)
        return decoder_output.predicted_ids[:, :, 0]

    def _rnn_cell(self, rnn_size=None, reuse=False):
        rnn_size = self.config['w2v_embedding_size'] if rnn_size is None else rnn_size
        return tf.nn.rnn_cell.GRUCell(rnn_size,
                                      kernel_initializer=tf.orthogonal_initializer(),
                                      reuse=reuse)

    def _bow_loss_fn(self, in_logits, in_targets):
         return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=in_targets, logits=in_logits), axis=-1)

    # sample-wise losses
    def _nll_loss_fn(self):
        mask_fn = lambda l: tf.sequence_mask(l, self.config['max_sequence_length'], dtype=tf.float32)
        mask = mask_fn(self.dec_seq_len)
        if (self.config['num_sampled'] == 0) or (self.config['vocabulary_size'] <= self.config['num_samples']):
            return tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                    targets=self.dec_out,
                                                    weights=mask,
                                                    average_across_timesteps=True,
                                                    average_across_batch=False)
        else:
            with tf.variable_scope('decoding/decoder/dense', reuse=True):
                mask = tf.reshape(mask, [-1])
                return mask * tf.nn.sampled_softmax_loss(weights=tf.transpose(tf.get_variable('kernel')),
                                                         biases=tf.get_variable('bias'),
                                                         labels=tf.reshape(self.dec_out, [-1, 1]),
                                                         inputs=tf.reshape(self.training_rnn_out, [-1, self.config['rnn_size']]),
                                                         num_sampled=self.config['num_sampled'],
                                                         num_classes=self.config['vocabulary_size'])

    def _kl_w_fn(self, anneal_max, anneal_bias, global_step):
        return anneal_max * tf.sigmoid((10 / anneal_bias) * (tf.cast(global_step, tf.float32) - tf.constant(anneal_bias / 2)))

    # sample-wise losses
    def _kl_loss_fn(self, mean, gamma):
        return 0.5 * tf.reduce_sum(tf.exp(gamma) + tf.square(mean) - 1 - gamma, axis=-1)

    def step(self, enc_inp, dec_inp, dec_out, bow_out, forward_only=False):
        dec_seq_length = np.ones(dec_out.shape[0]) * dec_out.shape[1]
        if forward_only:
            nll_loss, kl_w, kl_loss, bow_loss, overall_loss, batch_loss, step = self.session.run([self.nll_loss,
                                                                                        self.kl_w,
                                                                                        self.kl_loss,
                                                                                        self.bow_loss,
                                                                                        self.overall_loss,
                                                                                        self.batch_loss,
                                                                                        self.global_step],
                                                                                       feed_dict={self.enc_inp: enc_inp,
                                                                                                  self.dec_inp: dec_inp,
                                                                                                  self.dec_out: dec_out,
                                                                                                  self.bow_targets: bow_out,
                                                                                                  self.sequence_length_for_decoder: dec_seq_length})
        else:
            _, nll_loss, kl_w, kl_loss, bow_loss, overall_loss, batch_loss, step = self.session.run([self.train_op,
                                                                                           self.nll_loss,
                                                                                           self.kl_w,
                                                                                           self.kl_loss,
                                                                                           self.bow_loss,
                                                                                           self.overall_loss,
                                                                                           self.batch_loss,
                                                                                           self.global_step],
                                                                                          feed_dict={self.enc_inp: enc_inp,
                                                                                                     self.dec_inp: dec_inp,
                                                                                                     self.dec_out: dec_out,
                                                                                                     self.bow_targets: bow_out,
                                                                                                     self.sequence_length_for_decoder: dec_seq_length})
        return {'nll_loss': nll_loss,
                'kl_w': np.ones_like(kl_loss) * kl_w,
                'kl_loss': kl_loss,
                'bow_loss': bow_loss,
                'total_loss': overall_loss,
                'batch_loss': batch_loss,
                'step': step}

    def reconstruct(self, sentence, sentence_dropped):
        idx2word = self.rev_vocab
        print('I: %s' % ' '.join([idx2word[idx] for idx in sentence]))
        print()
        print('D: %s' % ' '.join([idx2word[idx] for idx in sentence_dropped]))
        print()
        predicted_ids = self.session.run(self.predicted_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        print('O: %s' % ' '.join([idx2word[idx] for idx in predicted_ids]))
        print('-'*12)

    def customized_reconstruct(self, sentence):
        idx2word = self.rev_vocab
        sentence = [self.get_new_w(w) for w in sentence.split()][:self.config['max_sequence_length']]
        print('I: %s' % ' '.join([idx2word[idx] for idx in sentence if idx != PAD_ID]))
        print()
        sentence = sentence + [PAD_ID] * (self.config['max_sequence_length'] - len(sentence))
        predicted_ids = self.session.run(self.predicted_ids, {self.enc_inp: np.atleast_2d(sentence)})[0]
        print('O: %s' % ' '.join([idx2word[idx] for idx in predicted_ids]))
        print('-'*12)

    def get_new_w(self, w):
        idx = self.vocab[w]
        return idx if idx < self.config['vocabulary_size'] else UNK_ID

    def generate(self):
        predicted_ids = self.session.run(self.predicted_ids,
                                         feed_dict={self._batch_size: 1,
                                                    self.z: np.random.randn(1, self.config['latent_size']),
                                                    self.enc_seq_len: [self.config['max_sequence_length']]})[0]
        print('G: %s' % ' '.join([self.rev_vocab[idx] for idx in predicted_ids]))
        print('-'*12)

    def _gradient_clipping(self, loss_op):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config['clip_norm'])
        return clipped_gradients, params

    def save(self, in_model_folder):
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.session, os.path.join(in_model_folder, W2VRAE.MODEL_NAME))
        with open(os.path.join(in_model_folder, W2VRAE.VOCAB_NAME), 'w') as vocab_out:
            print('\n'.join(self.rev_vocab), file=vocab_out)
        with open(os.path.join(in_model_folder, W2VRAE.CONFIG_NAME), 'w') as config_out:
            json.dump(self.config, config_out)

    @staticmethod
    def load(in_model_folder, in_session):
        with open(os.path.join(in_model_folder, W2VRAE.VOCAB_NAME), encoding='utf-8') as vocab_in:
            rev_vocab = list(map(lambda x: x.strip(), vocab_in))
        with open(os.path.join(in_model_folder, W2VRAE.CONFIG_NAME)) as config_in:
            config = json.load(config_in)
        ae = W2VRAE(config, rev_vocab, in_session, standalone=True)
        saver = tf.train.Saver()
        saver.restore(in_session, os.path.join(in_model_folder, W2VRAE.MODEL_NAME))
        return ae
