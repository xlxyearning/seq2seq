import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers.core import Dropout, Dense

import utils


def _build_multi_lstm_cell(num_units, num_layers, train_test_predict, keep_prob=1.0):
    cell = rnn_cell_impl.BasicLSTMCell(num_units, reuse=not (train_test_predict == 'train'))
    if train_test_predict == 'train' and keep_prob < 1.0:
        cell = rnn_cell_impl.DropoutWrapper(cell, output_keep_prob=keep_prob)
    cells = [cell for _ in range(num_layers)]
    return rnn_cell_impl.MultiRNNCell(cells)


class Encoder(object):
    """Implement the encoder model based on BasicLSTMCell, MultiRNNCell and dynamic_rnn

    """

    def __init__(self, source_input_ids, num_units, num_layers,
                 source_vocab_size, train_test_predict, keep_prob=1.0):
        """Construct the encoder, calculate the final states to be passed to the decoder.

        :param source_input_ids: a batch of inputs with shape (batch_size, time_steps)
        :param num_units: number of hidden units
        :param num_layers: number of MultiRNNCell layers
        :param source_vocab_size: size of source vocabulary
        :param train_test_predict: a str equals 'train', 'test', or 'predict'
        :param keep_prob: used in dropout layer
        """
        with tf.variable_scope('source_embedding_vector'):
            self.embedding_vector = tf.get_variable(name='source_embedding_vector',
                                                    shape=[source_vocab_size, num_units], dtype=tf.float32)

        embeded_input = tf.nn.embedding_lookup(self.embedding_vector, source_input_ids)
        if train_test_predict == 'train' and keep_prob < 1.0:
            embeded_input = tf.nn.dropout(embeded_input, keep_prob=keep_prob, name='input_dropout')
        _, self.final_states = self.encode(embeded_input, num_units, num_layers, train_test_predict)

    def encode(self, inputs, num_units, num_layers, train_test_predict):
        cell = _build_multi_lstm_cell(num_units, num_layers, train_test_predict)

        outputs, final_states = tf.nn.dynamic_rnn(cell=cell,
                                                  inputs=inputs,
                                                  dtype=tf.float32)
        return outputs, final_states


class Decoder(object):
    """Implement the decoder.

    """

    def __init__(self, train_test_predict, processed_target_input_ids, initial_state,
                 sequences_lengths, target_vocab_size, num_units, num_layers,
                 max_sequences_length, start_tokens=None, end_token=None):
        """Construct the decoder, calculate the final outputs.

        :param train_test_predict: a str equals 'train', 'test', or 'predict'
        :param processed_target_input_ids: shape [batch_size, time_steps], returned by utils.process_decoder_inputs
        :param initial_state: final state of encoder
        :param sequences_lengths: the true lengths of sequences without padding, shape [batch_size]
        :param target_vocab_size: target vocabulary size
        :param num_units: number of hidden units
        :param num_layers: number of layers
        :param max_sequences_length: the maximum iteration times
        :param start_tokens: shape [batch_size], used for inference,
        :param end_token: an integer, used for inference
        """
        with tf.variable_scope('target_embedding_vector'):
            self.embedding_vector = tf.get_variable(name='target_embedding_vector',
                                                    shape=[target_vocab_size, num_units],
                                                    dtype=tf.float32)
        embeded_input = tf.nn.embedding_lookup(self.embedding_vector, processed_target_input_ids)
        # choose different helper with respect to training or inference
        self.helper = self._helper(train_test_predict, embeded_input, sequences_lengths,
                                   start_tokens, end_token)

        self.cell = _build_multi_lstm_cell(num_units, num_layers, train_test_predict)

        self.output_layer = Dense(target_vocab_size)

        basic_decoder = seq2seq.BasicDecoder(cell=self.cell,
                                             helper=self.helper,
                                             initial_state=initial_state,
                                             output_layer=self.output_layer)

        self.final_outputs, self.final_state, self.final_sequence_lengths = \
            seq2seq.dynamic_decode(basic_decoder, impute_finished=True,
                                   maximum_iterations=max_sequences_length)

    def _helper(self, train_test_predict, embeded_inputs, sequences_lengths, start_tokens, end_token):
        if train_test_predict == 'train' or train_test_predict == 'test':
            helper = seq2seq.TrainingHelper(embeded_inputs, sequences_lengths)
        elif train_test_predict == 'predict':
            helper = seq2seq.GreedyEmbeddingHelper(self.embedding_vector, start_tokens, end_token)
        else:
            raise TypeError('train_test_predict should equals train, test, or predict')
        return helper


class Seq2seqModel(object):
    """Seq2seq_model is used to encode input to some state, then decode this intermediate

    to final outputs.
    """

    def __init__(self, source_input_ids, target_input_ids, num_units, num_layers,
                 source_vocab_size, target_vocab_size, target_sequences_lengths,
                 train_test_predict, grad_clip_norm, learning_rate, target_vocab_char_to_int,
                 batch_size, max_sequences_length, keep_prob=1.0, start_tokens=None, end_token=None):
        """Implement the Seq2seq_model

        :param source_input_ids: a batch of source inputs with shape (batch_size, source_time_steps)
        :param target_input_ids: a batch of target inputs with shape (batch_size, target_time_steps)
        :param num_units: number of hidden units
        :param num_layers: number of MultiRNNCell layers
        :param source_vocab_size: size of source vocabulary
        :param target_vocab_size: target vocabulary size
        :param target_sequences_lengths: the true lengths of target sequences without padding,
               shape [batch_size]
        :param train_test_predict: a str equals 'train', 'test', or 'predict'
        :param grad_clip_norm: to avoid gradients exploding problem
        :param learning_rate: learning_rate
        :param target_vocab_char_to_int: a dict mapping target characters to integers
        :param batch_size: batch size
        :param max_sequences_length: the maximum iteration times
        :param keep_prob: used in dropout layer
        :param start_tokens: an integer, used for inference, shape [batch_size]
        :param end_token: an integer, a scalar, used for inference,
        """
        with tf.name_scope('encoder'):
            my_encoder = Encoder(source_input_ids, num_units, num_layers,
                                 source_vocab_size, train_test_predict,
                                 keep_prob=keep_prob)

            self.decoder_initial_state = my_encoder.final_states

        with tf.name_scope('decoder'):
            processed_decoder_input_ids = utils.process_decoder_inputs(target_input_ids, batch_size,
                                                                       target_vocab_char_to_int)
            my_decoder = Decoder(train_test_predict, processed_decoder_input_ids,
                                 self.decoder_initial_state, target_sequences_lengths,
                                 target_vocab_size, num_units, num_layers,
                                 max_sequences_length, start_tokens, end_token)
            # final_outputs.run_output shape: (batch_size, target_time_step, target_vocab_size)
            # final_outputs.sample_id shape: (batch_size, target_time_step)
            self.logits = tf.identity(my_decoder.final_outputs.rnn_output, name='training_logits')
            self.sampled_ids = tf.identity(my_decoder.final_outputs.sample_id, name='inference_outputs')

        with tf.name_scope('optimization'):
            self.optimization(target_input_ids, self.logits, target_sequences_lengths,
                              max_sequences_length, grad_clip_norm, learning_rate)

    def optimization(self, target_output_ids, logits, target_sequences_lengths,
                     max_sequences_length, grad_clip_norm, learning_rate):
        with tf.name_scope('optimization'):
            mask = tf.sequence_mask(target_sequences_lengths, max_sequences_length, dtype=tf.float32,
                                    name='sequence_mask')
            self.loss = seq2seq.sequence_loss(logits, target_output_ids, mask)

            tf.summary.scalar('loss', self.loss)

            optimizer = tf.train.AdamOptimizer(learning_rate)

            var_list = tf.trainable_variables()
            # grad = optimizer.compute_gradients(self.loss, var_list)
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, var_list), grad_clip_norm, name='clip_norm')
            self.train_op = optimizer.apply_gradients(zip(grads, var_list))
