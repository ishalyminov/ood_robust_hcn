import sys

import numpy as np

sys.path.append('..')

from utils.training_utils import batch_generator


def get_kl_loss_stats(in_model, in_dataset, in_session, batch_size=64):
    batch_gen = batch_generator(in_dataset, batch_size)
    losses = []
    for batch_encoder_input, batch_decoder_input, batch_decoder_output in batch_gen:
        batch_loss_dict, batch_outputs = in_model.step(batch_encoder_input, batch_decoder_input, batch_decoder_output, in_session, forward_only=True)
        losses += batch_loss_dict['kl_loss'].tolist()
    return {'max': np.max(losses), 'min': np.min(losses), 'avg': np.mean(losses)}


class VAEOODDetector(object):
    def __init__(self, in_vae, loss_threshold=0):
        self.vae = in_vae
        self.threshold = loss_threshold

    def tune_threshold(self, in_dataset, in_session, in_decision_type):
        dataset_stats = get_kl_loss_stats(self.vae, in_dataset, in_session)
        self.threshold = dataset_stats[in_decision_type]

    def predict(self, inputs, in_session):
        encoder_input, decoder_input, decoder_output = inputs
        loss_dict, outputs = self.vae.step(encoder_input, decoder_input, decoder_output, in_session, forward_only=True)
        return loss_dict['kl_loss'] < self.threshold

    def predict_loss(self, inputs, in_session):
        encoder_input, decoder_input, decoder_output = inputs
        loss_dict, outputs = self.ae.step(encoder_input, decoder_input, decoder_output, in_session, forward_only=True)
        return loss_dict['kl_loss'], loss_dict['kl_loss'] < self.threshold
