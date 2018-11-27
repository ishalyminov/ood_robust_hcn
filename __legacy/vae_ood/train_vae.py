import json
import os
from argparse import ArgumentParser
import sys
import random

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rnn_vae import RNNVAE
from utils.preprocessing import EOS, UNK_ID, make_vocabulary, make_autoencoder_dataset, load_txt
from utils.training_utils import batch_generator

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)


def train(in_session, in_model, in_train, in_dev, in_dst_folder, nb_epoch, batch_size, early_stopping_threshold, dropout_keep_prob, **kwargs):
    best_dev_loss = np.inf
    epochs_without_improvement = 0
    for epoch_counter in range(nb_epoch):
        [random.shuffle(train_i) for train_i in in_train]
        batch_gen = batch_generator(in_train, batch_size, verbose=False)
        train_nll_losses = []
        train_kl_losses = []
        train_kl_w = []
        for idx, (batch_enc_input, batch_dec_input, batch_dec_output) in enumerate(batch_gen):
            dec_input_dropped = [token
                                 if random.random() < dropout_keep_prob else UNK_ID
                                 for token in batch_dec_input[:, 1:].flatten()]
            batch_dec_input_dropped = np.concatenate([np.expand_dims(batch_dec_input[:,0], axis=-1), np.array(dec_input_dropped).reshape(batch_dec_input.shape[0], batch_dec_input.shape[1] - 1)], axis=1)
            loss_dict = in_model.step(batch_enc_input, batch_dec_input_dropped, batch_dec_output, in_session)
            train_nll_losses.append(loss_dict['nll_loss'])
            train_kl_losses.append(loss_dict['kl_loss'])
            train_kl_w.append(loss_dict['kl_w'])
        train_loss = np.mean(np.array(train_nll_losses) + np.array(train_kl_losses) * np.array(train_kl_w))
        print('Epoch {} out of {} results'.format(epoch_counter, nb_epoch))
        print('train loss: {:.3f} | nll_loss: {:.3f} | kl_loss: {:.3f} | kl_w: {:.5f}'.format(train_loss,
                                                                                              np.mean(train_nll_losses),
                                                                                              np.mean(train_kl_losses),
                                                                                              loss_dict['kl_w']))
        dev_eval = evaluate(in_session,
                            in_model,
                            in_dev)
        print('dev loss: {:.3f} | nll_loss: {:.3f} | kl_loss: {:.3f} | kl_w: {:.5f}'.format(dev_eval['loss'],
                                                                                            dev_eval['nll_loss'],
                                                                                            dev_eval['kl_loss'],
                                                                                            dev_eval['kl_w']))
        if dev_eval['loss'] < best_dev_loss:
            best_dev_loss = dev_eval['loss'] 
            in_model.save(in_dst_folder, in_session)
            print('New best loss. Saving checkpoint')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if early_stopping_threshold < epochs_without_improvement:
            print('Early stopping after {} epochs'.format(epoch_counter))
            break
    print('Optimization Finished!')


def evaluate(in_session, in_model, in_dataset, batch_size=64):
    batch_gen = batch_generator(in_dataset, batch_size)
    kl_losses = []
    nll_losses = []
    kl_w = []
    outputs = []
    for batch_enc_input, batch_dec_input, batch_dec_output in batch_gen:
        loss_dict, batch_outputs = in_model.step(batch_enc_input, batch_dec_input, batch_dec_output, in_session, forward_only=True)
        kl_losses.append(loss_dict['kl_loss'])
        nll_losses.append(loss_dict['nll_loss'])
        kl_w.append(loss_dict['kl_w'])
        outputs += list(batch_outputs)
    print('10 random eval sequences:')
    random_idx = np.random.choice(range(len(outputs)), size=10)
    for idx in random_idx:
        output = list(outputs[idx])
        if EOS in output:
            output = output[:output.index(EOS)]
        print(' '.join(output))
    loss = np.mean(np.array(kl_losses) * np.array(kl_w) + np.array(nll_losses))
    ppx = np.exp(loss)
    return {'nll_loss': np.mean(nll_losses), 'kl_loss': np.mean(kl_losses), 'loss': loss, 'perplexity': ppx, 'kl_w': loss_dict['kl_w']}


def main(in_trainset_file, in_devset_file, in_testset_file, in_model_folder, in_config_file):
    with open(in_config_file) as config_in:
        config = json.load(config_in)
    train_utterances = load_txt(in_trainset_file)
    dev_utterances = load_txt(in_devset_file)
    test_utterances = load_txt(in_testset_file)

    vocab, rev_vocab = make_vocabulary(train_utterances, config['max_vocabulary_size'])
    config['vocabulary_size'] = len(vocab)
    train_data = make_autoencoder_dataset(train_utterances, vocab, config['max_sequence_length'])
    dev_data = make_autoencoder_dataset(dev_utterances, vocab, config['max_sequence_length'])
    test_data = make_autoencoder_dataset(test_utterances, vocab, config['max_sequence_length'])

    with tf.Session() as sess:
        ae = RNNVAE(config, rev_vocab)
        sess.run(tf.global_variables_initializer())
        train(sess, ae, train_data, dev_data, in_model_folder, **config)


def configure_argument_parser():
    result_parser = ArgumentParser(description='Train a VAE model on a text corpus')
    result_parser.add_argument('trainset')
    result_parser.add_argument('devset')
    result_parser.add_argument('testset')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('--config', help='config JSON file')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    main(args.trainset, args.devset, args.testset, args.model_folder, args.config)

