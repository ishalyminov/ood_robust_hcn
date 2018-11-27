from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import _beam_search_step
import tensorflow as tf
import numpy as np


class ModifiedBasicDecoder(tf.contrib.seq2seq.BasicDecoder):
    def __init__(self, cell, helper, initial_state, concat_z, output_layer=None):
        super().__init__(cell, helper, initial_state)
        self.z = concat_z

    def initialize(self, name=None):
        (finished, first_inputs, initial_state) =  self._helper.initialize() + (self._initial_state,)
        first_inputs = tf.concat([first_inputs, self.z], -1)
        return (finished, first_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        with tf.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
        if self._output_layer is not None:
            cell_outputs = self._output_layer(cell_outputs)
        sample_ids = self._helper.sample(
            time=time, outputs=cell_outputs, state=cell_state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=cell_outputs,
            state=cell_state,
            sample_ids=sample_ids)
        outputs = tf.contrib.seq2seq.BasicDecoderOutput(cell_outputs, sample_ids)
        next_inputs = tf.concat([next_inputs, self.z], -1)
        return (outputs, next_state, next_inputs, finished)


class ModifiedBeamSearchDecoder(tf.contrib.seq2seq.BeamSearchDecoder):
    def __init__(self,
                 cell,
                 embedding,
                 start_tokens,
                 end_token,
                 initial_state,
                 beam_width,
                 concat_z,
                 output_layer=None,
                 length_penalty_weight=0.0):
        super().__init__(cell, embedding, start_tokens, end_token, initial_state, beam_width, output_layer, length_penalty_weight)
        self.z = concat_z

    def initialize(self, name=None):
        finished, start_inputs = self._finished, self._start_inputs

        start_inputs = tf.concat([start_inputs, self.z], -1)

        log_probs = tf.one_hot(  # shape(batch_sz, beam_sz)
            tf.zeros([self._batch_size], dtype=tf.int32),
            depth=self._beam_width,
            on_value=0.0,
            off_value=-np.Inf,
            dtype=nest.flatten(self._initial_cell_state)[0].dtype)

        initial_state = tf.contrib.seq2seq.BeamSearchDecoderState(
            cell_state=self._initial_cell_state,
            log_probs=log_probs,
            finished=finished,
            lengths=tf.zeros(
                [self._batch_size, self._beam_width], dtype=tf.int64))

        return (finished, start_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight

        with tf.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state,
                                            self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            beam_search_output, beam_search_state = _beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = tf.cond(
                tf.reduce_all(finished), lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

            next_inputs = tf.concat([next_inputs, self.z], -1)

        return (beam_search_output, beam_search_state, next_inputs, finished)

