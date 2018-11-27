from argparse import ArgumentParser
import sys

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append('..')

from ae_ood import AEOODDetector
from compatible_rnn_autoencoder import CompatibleRNNAutoencoder
from utils.preprocessing import make_variational_autoencoder_dataset, load_txt
from utils.training_utils import batch_generator


def evaluate(in_session, in_model, in_dataset, batch_size=64):
    X, masks, labels = in_dataset
    batch_gen = batch_generator((X, masks), batch_size)
    predictions = []
    for batch in batch_gen:
        batch_predictions = in_model.predict(batch, in_session)
        predictions += batch_predictions.tolist()
    print(confusion_matrix(labels, predictions))
    return accuracy_score(labels, predictions)


def main(in_model_folder, in_devset_file, in_evalset_file, in_decision_type):
    dev_utterances = load_txt(in_devset_file)
    evalset = pd.read_json(in_evalset_file)

    eval_utterances = list(map(lambda x: x.lower().split(), evalset.utterance))
    with tf.Session() as sess:
        ae = CompatibleRNNAutoencoder.load(in_model_folder, sess)
        rev_vocab, config = ae.vocab, ae.config
        vocab = {word: idx for idx, word in enumerate(rev_vocab)}
        dev_enc_inp, _, dev_dec_out, _ = make_variational_autoencoder_dataset(dev_utterances, vocab, config['max_sequence_length'])
        eval_enc_inp, _, eval_dec_out, _ = make_variational_autoencoder_dataset(eval_utterances, vocab, config['max_sequence_length'])
        ae_ood = AEOODDetector(ae)
        ae_ood.tune_threshold((dev_enc_inp, dev_dec_out), sess, in_decision_type)
        print('Detector accuracy on the evalset: {:.3f}'.format(evaluate(sess, ae_ood, (eval_enc_inp, eval_dec_out, evalset.label))))


def configure_argument_parser():
    result_parser = ArgumentParser(description='Evaluate the autoencoder for OOD utterance detection')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('devset', help='in-domain data the AE was fitted on [.txt]')
    result_parser.add_argument('evalset', help='in-domain/out-of-domain evaluation data [.json]')
    result_parser.add_argument('--decision_type',
                               help='Reconstruction loss value threshold from the devset upon which outliers are told [min/max/avg]',
                               default='max')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    main(args.model_folder, args.devset, args.evalset, args.decision_type)

