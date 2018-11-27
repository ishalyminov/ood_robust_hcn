import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.training_utils import batch_generator


def get_loss_stats(in_model, in_dataset, loss_components=['kl_loss', 'nll_loss'], batch_size=64):
    batch_gen = batch_generator(in_dataset, batch_size)
    losses = []
    for batch in batch_gen:
        loss_dict = in_model.step(*batch, forward_only=True)
        cumulative_loss = np.sum([loss_dict[component] for component in loss_components])
        losses.append(cumulative_loss)
    return {'max': np.max(losses), 'min': np.min(losses), 'avg': np.mean(losses)}


class VAEOODDetector(object):
    def __init__(self, in_vae, loss_threshold=0, loss_components=['nll_loss', 'kl_loss']):
        self.vae = in_vae
        self.threshold = loss_threshold
        self.loss_components = loss_components

    def tune_threshold(self, in_dataset, in_decision_type):
        dataset_stats = get_loss_stats(self.vae, in_dataset, loss_components=self.loss_components)
        self.threshold = dataset_stats[in_decision_type]

    def predict(self, inputs):
        loss_dict = self.vae.step(*inputs, forward_only=True)
        losses = np.sum([loss_dict[component] for component in self.loss_components], axis=0)
        return losses < self.threshold

    def predict_loss(self, inputs):
        loss_dict = self.vae.step(*inputs, forward_only=True)
        losses = np.sum([loss_dict[component] for component in self.loss_components], axis=0)
        return losses, losses < self.threshold

