from argparse import ArgumentParser
import sys

import numpy as np
import tensorflow as tf
import pandas as pd

sys.path.append('..')

from compatible_rnn_autoencoder import CompatibleRNNAutoencoder
from utils.preprocessing import make_variational_autoencoder_dataset, load_txt
from utils.training_utils import batch_generator


def run_autoencoder(in_session, in_model, in_dataset, batch_size=64):
    X, masks, labels = in_dataset
    batch_gen = batch_generator((X, masks), batch_size)
    predictions = []
    for batch in batch_gen:
        batch_predictions = in_model.predict(batch, in_session)
        predictions += batch_predictions.tolist()
    return predictions


def main(in_model_folder, in_dataset_file, in_result_file):
    dev_utterances = load_txt(in_dataset_file)

    utterances = list(map(lambda x: x.lower().split(), evalset.utterance))
    with tf.Session() as sess:
        ae = CompatibleRNNAutoencoder.load(in_model_folder, sess)
        rev_vocab, config = ae.vocab, ae.config
        vocab = {word: idx for idx, word in enumerate(rev_vocab)}
        enc_inp, _, _, _ = make_variational_autoencoder_dataset(utterances, vocab, config['max_sequence_length'])
        reconstruction_scores = run_autoencoder(sess, ae, enc_inp)
    assert len(reconstruction_scores) == len(utterances)
    result_df = pd.DataFrame({'utterance': evalset.utterance, 'ae_reconstruction_score': reconstruction_scores})
    result_df.to_json(in_result_file)

def configure_argument_parser():
    result_parser = ArgumentParser(description='Run autoencoder reconstruction scores')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('dataset', help='utterances to  [.txt]')
    result_parser.add_argument('result_file', help='output file [.json]')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    main(args.model_folder, args.devset, args.evalset, args.decision_type)
