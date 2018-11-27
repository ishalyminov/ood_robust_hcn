from argparse import ArgumentParser
import sys

import tensorflow as tf
import pandas as pd

sys.path.append('..')

from ae_ood import AEOODDetector
from rnn_autoencoder import RNNAutoencoder
from utils.preprocessing import make_dataset, load_txt
from utils.training_utils import batch_generator


def predict(in_session, in_model, in_dataset, batch_size=64):
    X, masks, labels = in_dataset
    batch_gen = batch_generator((X, masks), batch_size)
    losses, predictions, = [], []
    for batch_x, batch_masks in batch_gen:
        batch_predictions = in_model.predict_loss(batch_x, batch_masks, in_session)
        losses += batch_predictions[0].tolist()
        predictions += batch_predictions[1].tolist()
    return losses, predictions


def main(in_model_folder, in_devset_file, in_testset_file, in_decision_type):
    dev_utterances = load_txt(in_devset_file)
    testset = pd.read_json(in_testset_file)

    test_utterances = list(map(lambda x: x.split(), testset.utterance))
    with tf.Session() as sess:
        ae = RNNAutoencoder.load(in_model_folder, sess)
        rev_vocab, config = ae.vocab, ae.config
        vocab = {word: idx for idx, word in enumerate(rev_vocab)}
        dev_X, dev_masks = make_dataset(dev_utterances, vocab, config['max_sequence_length'])
        test_X, test_masks = make_dataset(test_utterances, vocab, config['max_sequence_length'])
        ae_ood = AEOODDetector(ae)
        ae_ood.tune_threshold((dev_X, dev_masks), sess, in_decision_type)
        print('Decision threshold: {:.3f}'.format(ae_ood.threshold))
        print('Utterance\tloss\tprediction')
        losses, predictions = predict(sess, ae_ood, (test_X, test_masks, testset.label))
        for utterance, loss, prediction in zip(test_utterances, losses, predictions):
            print('{}\t{:.3f}\t{}'.format(' '.join(utterance), loss, prediction))


def configure_argument_parser():
    result_parser = ArgumentParser(description='Run the autoencoder for OOD utterance detection')
    result_parser.add_argument('model_folder')
    result_parser.add_argument('devset', help='in-domain data the AE was fitted on [.txt]')
    result_parser.add_argument('testset', help='in-domain/out-of-domain evaluation data [.json]')
    result_parser.add_argument('--decision_type',
                               help='Reconstruction loss value threshold from the devset upon which outliers are told [min/max/avg]',
                               default='max')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    main(args.model_folder, args.devset, args.testset, args.decision_type)

