import tensorflow as tf

import utils


LEARNING_RATE = 1e-2
BATCH_SIZE = 128
NUM_LAYERS = 2
NUM_UNITS = 50
NUM_EPOCHS = 60
GRAD_CLIP_NORM = 5.0


source_data_list, target_data_list \
    = utils.read_data('./data/letters_source.txt', './data/letters_target.txt')

# Construct the vocabularies
source_vocab_char_to_int, source_vocab_int_to_char, \
target_vocab_char_to_int, target_vocab_int_to_char = utils.construct_character_vocab(source_data_list,
                                                                                     target_data_list)


def input_to_int(input, source_vocab_char_to_int):
    return [source_vocab_char_to_int[ch] for ch in input]


input_word = 'helloboygirl'
input_word_int = input_to_int(input_word, source_vocab_char_to_int)
# print([input_word_int] * BATCH_SIZE)
input_word_int_batch = [input_word_int] * BATCH_SIZE

graph = tf.Graph()

checkpoint = './checkpoint/seq2seq.ckpt'
# saver = tf.train.Saver()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    saver.restore(sess, checkpoint)

    predict_output_op = sess.graph.get_tensor_by_name('Predict/Model/decoder/inference_outputs:0')
    batch_source_input_ids = sess.graph.get_tensor_by_name('batch_source_input_ids:0')
    batch_target_input_ids = sess.graph.get_tensor_by_name('batch_target_input_ids:0')
    target_sequences_lengths = sess.graph.get_tensor_by_name('target_sequences_lengths:0')

    feed_dict = {batch_source_input_ids: input_word_int_batch,
                 batch_target_input_ids: input_word_int_batch,
                 target_sequences_lengths: [len(input_word_int)+3]*BATCH_SIZE}

    output_ids = sess.run([predict_output_op], feed_dict=feed_dict)[0]

    print('Source input: ')
    print('input_word:     ' + input_word)
    print('input_word_ids: ' + str(input_word_int))
    print('Output: ')
    print('output_word:    ' + str([target_vocab_int_to_char[i] for i in output_ids[0]]))
    print('output_word_ids:' + str(output_ids[0]))
    print(len(input_word))
    print(len(output_ids[0]))
