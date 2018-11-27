from __future__ import print_function

from argparse import ArgumentParser
import json
import sys
import os
import random

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vrae import VRAE
from utils.preprocessing import load_txt, make_vocabulary, make_variational_autoencoder_dataset
from utils.training_utils import batch_generator


random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)


def train(in_model, in_train, in_dev, in_config, in_model_folder):
    best_dev_loss = np.inf
    epochs_without_improvement = 0
    for epoch in range(in_config['num_epoch']):
        print("Epoch [%d/%d]" % (epoch + 1, in_config['num_epoch']))
        batch_gen = batch_generator(in_train, in_config['batch_size'])
        for batch in batch_gen:
            in_model.step(*batch)
        trn_loss_dict = evaluate(in_model, in_train)
        print("trn results | nll_loss:%.3f | kl_w:%.3f | kl_loss:%.3f | bow_loss:%.3f" % (trn_loss_dict['nll_loss'],
                                                                                          trn_loss_dict['kl_w'],
                                                                                          trn_loss_dict['kl_loss'],
                                                                                          trn_loss_dict['bow_loss']))
        dev_loss_dict = evaluate(in_model, in_dev)
        print("dev results | nll_loss:%.3f | kl_w:%.3f | kl_loss:%.3f | bow_loss:%.3f" % (dev_loss_dict['nll_loss'],
                                                                                          dev_loss_dict['kl_w'],
                                                                                          dev_loss_dict['kl_loss'],
                                                                                          dev_loss_dict['bow_loss']))
        for idx in np.random.choice(np.arange(len(in_dev)), size=5):
            encoder_input, decoder_input, decoder_output, bow_output = [dev_i[idx] for dev_i in in_dev]
            encoder_input = ' '.join([in_model.rev_vocab[word] for word in encoder_input])
            in_model.customized_reconstruct(encoder_input)
        resulting_dev_loss = dev_loss_dict['loss']
        if resulting_dev_loss < best_dev_loss:
            print('New best dev loss! Saving checkpoint')
            best_dev_loss = resulting_dev_loss
            in_model.save(in_model_folder)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement == in_config['early_stopping_threshold']:
            print('Stopped after {} epochs due to no loss improvement'.format(epoch + 1))
            break


def evaluate(in_model, in_dataset, batch_size=64):
    batch_gen = batch_generator(in_dataset, batch_size)
    kl_losses = []
    nll_losses = []
    kl_w = []
    bow_losses = []
    for batch in batch_gen:
        loss_dict = in_model.step(*batch, forward_only=True)
        kl_losses += loss_dict['kl_loss'].tolist()
        nll_losses += loss_dict['nll_loss'].tolist()
        kl_w += loss_dict['kl_w'].tolist()
        bow_losses += loss_dict['bow_loss'].tolist()
    loss = np.mean(np.array(kl_losses) * np.array(kl_w) + np.array(nll_losses))
    return {'loss': loss,
            'nll_loss': np.mean(nll_losses),
            'kl_loss': np.mean(kl_losses),
            'kl_w': np.mean(loss_dict['kl_w']),
            'bow_loss': np.mean(bow_losses)}


def main(in_trainset_file, in_devset_file, in_testset_file, in_config, in_model_folder):
    train_utterances, dev_utterances, test_utterances = load_txt(in_trainset_file), load_txt(in_devset_file), load_txt(in_testset_file)

    vocab, rev_vocab = make_vocabulary(train_utterances,
                                       in_config['max_vocabulary_size'],
                                       frequency_threshold=0,
                                       ngram_sizes=(1,))
    config['vocabulary_size'] = len(vocab)

    train_X = make_variational_autoencoder_dataset(train_utterances, vocab, config['max_sequence_length'])
    dev_X = make_variational_autoencoder_dataset(dev_utterances, vocab, config['max_sequence_length'])
    test_X = make_variational_autoencoder_dataset(test_utterances, vocab, config['max_sequence_length'])

    # save_model(vocab, config, in_model_folder)
    with tf.Session() as sess:
        model = VRAE(config, rev_vocab, sess, standalone=True)
        train(model, train_X, dev_X, config, in_model_folder)


def configure_argument_parser():
    result_parser = ArgumentParser(description='Train an VAE model on a text corpus')
    result_parser.add_argument('trainset')
    result_parser.add_argument('devset')
    result_parser.add_argument('testset')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('--config', help='config JSON file', default='vae.json')
    return result_parser


if __name__ == '__main__':
     parser = configure_argument_parser()
     args = parser.parse_args()
     with open(args.config) as config_in:
        config = json.load(config_in)
     main(args.trainset, args.devset, args.testset, config, args.model_folder)

