import sys

import numpy as np

sys.path.append('..')

from utils.training_utils import batch_generator


def get_loss_stats(in_model, in_dataset, in_session, batch_size=64):
    batch_gen = batch_generator(in_dataset, batch_size)
    losses = []
    for batch in batch_gen:
        batch_losses, batch_outputs = in_model.step(*batch, in_session, forward_only=True)
        losses += list(batch_losses)
    return {'max': np.max(losses), 'min': np.min(losses), 'avg': np.mean(losses)}


class AEOODDetector(object):
    def __init__(self, in_ae, loss_threshold=0):
        self.ae = in_ae
        self.threshold = loss_threshold

    def tune_threshold(self, in_dataset, in_session, in_decision_type):
        dataset_stats = get_loss_stats(self.ae, in_dataset, in_session)
        self.threshold = dataset_stats[in_decision_type]

    def predict(self, X, in_session):
        batch_losses, outputs = self.ae.step(*X, in_session, forward_only=True)
        return batch_losses < self.threshold

    def predict_loss(self, X, in_session):
        batch_losses, outputs = self.ae.step(*X, in_session, forward_only=True)
        return batch_losses, batch_losses < self.threshold
