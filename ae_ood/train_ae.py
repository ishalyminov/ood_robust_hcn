import json
import os
from argparse import ArgumentParser
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from compatible_rnn_autoencoder import CompatibleRNNAutoencoder
from utils.preprocessing import PAD, START, UNK, EOS, make_vocabulary, make_variational_autoencoder_dataset, load_txt
from utils.training_utils import batch_generator


def train(in_session, in_model, in_train, in_dev, in_dst_folder, nb_epoch, batch_size, early_stopping_threshold, **kwargs):
    best_dev_loss = np.inf
    epochs_without_improvement = 0
    for epoch_counter in range(nb_epoch):
        batch_gen = batch_generator(in_train, batch_size)
        train_batch_losses = []
        for batch in batch_gen:
            enc_inp, dec_out = batch
            train_batch_loss = in_model.step(enc_inp, dec_out, in_session)
            train_batch_losses.append(train_batch_loss)
        train_loss = np.mean(train_batch_losses)
        print('Epoch {} out of {} results'.format(epoch_counter, nb_epoch))
        print('train loss: {:.3f}'.format(train_loss))
        dev_eval = evaluate(in_session,
                            in_model,
                            in_dev)
        print('; '.join(['dev {}: {:.3f}'.format(key, value)
                         for key, value in dev_eval.items()]))
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
    losses = []
    outputs = []
    for batch in batch_gen:
        enc_inp, dec_out = batch
        batch_losses, batch_outputs = in_model.step(enc_inp, dec_out, in_session, forward_only=True)
        losses += list(batch_losses)
        outputs += list(batch_outputs)
    print('10 random eval sequences:')
    random_idx = np.random.choice(range(len(outputs)), size=10)
    for idx in random_idx:
        output = list(outputs[idx])
        if EOS in output:
            output = output[:output.index(EOS)]
        print(' '.join(output))
    loss = np.mean(losses)
    ppx = np.exp(loss)
    return {'loss': loss, 'perplexity': ppx}


def main(in_trainset_file, in_devset_file, in_testset_file, in_model_folder, in_config_file, in_custom_vocab):
    with open(in_config_file) as config_in:
        config = json.load(config_in)
    train_utterances = load_txt(in_trainset_file)
    dev_utterances = load_txt(in_devset_file)
    test_utterances = load_txt(in_testset_file)

    if in_custom_vocab is not None:
        with open(in_custom_vocab) as vocab_in:
            rev_vocab = [line.rstrip() for line in vocab_in]
            vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    else:
        vocab, rev_vocab = make_vocabulary(train_utterances,
                                           config['max_vocabulary_size'],
                                           special_tokens=[PAD, START, UNK, EOS])
    config['vocabulary_size'] = len(vocab)

    train_enc_inp, _, train_dec_out, _ = make_variational_autoencoder_dataset(train_utterances, vocab, config['max_sequence_length'])
    dev_enc_inp, _, dev_dec_out, _ = make_variational_autoencoder_dataset(dev_utterances, vocab, config['max_sequence_length'])
    test_enc_inp, _, test_dec_out, _ = make_variational_autoencoder_dataset(test_utterances, vocab, config['max_sequence_length'])

    with tf.Session() as sess:
        ae = CompatibleRNNAutoencoder(config, rev_vocab)
        sess.run(tf.global_variables_initializer())
        train(sess, ae, (train_enc_inp, train_dec_out), (dev_enc_inp, dev_dec_out), in_model_folder, **config)


def configure_argument_parser():
    result_parser = ArgumentParser(description='Train an autoencoder model on a text corpus')
    result_parser.add_argument('trainset')
    result_parser.add_argument('devset')
    result_parser.add_argument('testset')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('--config', help='config JSON file')
    result_parser.add_argument('--custom_vocab', default=None, type=str)
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    main(args.trainset, args.devset, args.testset, args.model_folder, args.config, args.custom_vocab)

