import os
from argparse import ArgumentParser
import sys

import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vae_ood import VAEOODDetector
from vrae import VRAE
from utils.preprocessing import make_variational_autoencoder_dataset, load_txt
from utils.training_utils import batch_generator


def evaluate(in_model, in_dataset, batch_size=64):
    data, labels = in_dataset[:-1], in_dataset[-1]
    batch_gen = batch_generator(data, batch_size)
    predictions = []
    for batch in batch_gen:
        batch_predictions = in_model.predict(batch)
        predictions += list(batch_predictions)
    print(confusion_matrix(labels, predictions))
    return accuracy_score(labels, predictions)


def main(in_model_folder, in_devset_file, in_evalset_file, in_decision_type, in_loss_components):
    dev_utterances = load_txt(in_devset_file)
    evalset = pd.read_json(in_evalset_file)

    eval_utterances = list(map(lambda x: x.split(), evalset.utterance))
    with tf.Session() as sess:
        vae = VRAE.load(in_model_folder, sess)
        rev_vocab, config = vae.vocab, vae.config
        vocab = {word: idx for idx, word in enumerate(rev_vocab)}
        train_data = make_variational_autoencoder_dataset(dev_utterances, vocab, config['max_sequence_length'])
        eval_data = make_variational_autoencoder_dataset(eval_utterances, vocab, config['max_sequence_length'])
        vae_ood = VAEOODDetector(vae, loss_components=in_loss_components)
        vae_ood.tune_threshold(train_data, in_decision_type)
        print('Detector accuracy on the evalset: {:.3f}'.format(evaluate(vae_ood, (*eval_data, evalset.label))))


def configure_argument_parser():
    result_parser = ArgumentParser(description='Evaluate the autoencoder for OOD utterance detection')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('devset', help='in-domain data the AE was fitted on [.txt]')
    result_parser.add_argument('evalset', help='in-domain/out-of-domain evaluation data [.json]')
    result_parser.add_argument('--loss_components',
                               help='kl_loss and/or nll_loss (,-separated)',
                               default='kl_loss,nll_loss')
    result_parser.add_argument('--decision_type',
                               help='Loss value threshold from the devset upon which outliers are told [min/max/avg]',
                               default='max')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    loss_components = [component.strip() for component in args.loss_components.split(',')]
    main(args.model_folder, args.devset, args.evalset, args.decision_type, loss_components)
