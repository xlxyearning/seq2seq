import tensorflow as tf

import seq2seq_model
import utils

LEARNING_RATE = 1e-2
BATCH_SIZE = 128
NUM_LAYERS = 2
NUM_UNITS = 50
NUM_EPOCHS = 60
GRAD_CLIP_NORM = 5.0

checkpoint_dir = './checkpoint/'

# Construct the vocabularies
source_data_list, target_data_list \
    = utils.read_data('./data/letters_source.txt', './data/letters_target.txt')
source_vocab_char_to_int, source_vocab_int_to_char, \
    target_vocab_char_to_int, target_vocab_int_to_char = utils.construct_character_vocab(source_data_list,
                                                                                         target_data_list)

source_int, target_int = utils.source_target_to_int(source_data_list, target_data_list,
                                                    source_vocab_char_to_int, target_vocab_char_to_int)

train_source_int = source_int[BATCH_SIZE:]
train_target_int = target_int[BATCH_SIZE:]
test_source_int = source_int[: BATCH_SIZE]
test_target_int = target_int[: BATCH_SIZE]
test_generator = utils.get_batch(BATCH_SIZE, test_source_int, test_target_int,
                                 source_vocab_char_to_int['<PAD>'], target_vocab_char_to_int['<PAD>'])
test_pad_source, test_pad_target, test_target_lengths, test_source_lengths = next(test_generator)

graph = tf.Graph()
with graph.as_default():

    batch_source_input_ids, batch_target_input_ids, \
        target_sequences_lengths, max_target_sequences_length = utils.get_inputs()

    with tf.name_scope('Train'):
        with tf.variable_scope('Model'):
            train_model = seq2seq_model.Seq2seqModel(source_input_ids=batch_source_input_ids,
                                                     target_input_ids=batch_target_input_ids,
                                                     num_units=NUM_UNITS,
                                                     num_layers=NUM_LAYERS,
                                                     source_vocab_size=len(source_vocab_char_to_int),
                                                     target_vocab_size=len(target_vocab_char_to_int),
                                                     target_sequences_lengths=target_sequences_lengths,
                                                     train_test_predict='train',
                                                     grad_clip_norm=GRAD_CLIP_NORM,
                                                     learning_rate=LEARNING_RATE,
                                                     target_vocab_char_to_int=target_vocab_char_to_int,
                                                     keep_prob=0.7,
                                                     batch_size=BATCH_SIZE,
                                                     max_sequences_length=max_target_sequences_length)
    with tf.name_scope('Test'):
        with tf.variable_scope('Model', reuse=True):
            test_model = seq2seq_model.Seq2seqModel(source_input_ids=batch_source_input_ids,
                                                    target_input_ids=batch_target_input_ids,
                                                    num_units=NUM_UNITS,
                                                    num_layers=NUM_LAYERS,
                                                    source_vocab_size=len(source_vocab_char_to_int),
                                                    target_vocab_size=len(target_vocab_char_to_int),
                                                    target_sequences_lengths=target_sequences_lengths,
                                                    train_test_predict='test',
                                                    grad_clip_norm=GRAD_CLIP_NORM,
                                                    learning_rate=LEARNING_RATE,
                                                    target_vocab_char_to_int=target_vocab_char_to_int,
                                                    keep_prob=1.0,
                                                    batch_size=BATCH_SIZE,
                                                    max_sequences_length=max_target_sequences_length)
    with tf.name_scope('Predict'):
        with tf.variable_scope('Model', reuse=True):
            start_tokens = tf.tile(tf.constant([target_vocab_char_to_int['<GO>']], dtype=tf.int32), [BATCH_SIZE])
            end_token = tf.constant(target_vocab_char_to_int['<EOS>'], dtype=tf.int32)
            predict_model = seq2seq_model.Seq2seqModel(source_input_ids=batch_source_input_ids,
                                                       target_input_ids=batch_target_input_ids,
                                                       num_units=NUM_UNITS,
                                                       num_layers=NUM_LAYERS,
                                                       source_vocab_size=len(source_vocab_char_to_int),
                                                       target_vocab_size=len(target_vocab_char_to_int),
                                                       target_sequences_lengths=target_sequences_lengths,
                                                       train_test_predict='predict',
                                                       grad_clip_norm=GRAD_CLIP_NORM,
                                                       learning_rate=LEARNING_RATE,
                                                       target_vocab_char_to_int=target_vocab_char_to_int,
                                                       batch_size=BATCH_SIZE,
                                                       keep_prob=1.0,
                                                       max_sequences_length=max_target_sequences_length,
                                                       start_tokens=start_tokens, end_token=end_token)

    merged_summary = tf.summary.merge_all()

    file_writer = tf.summary.FileWriter(logdir='./logdir/', graph=graph)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    # restore parameters to continue to train
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init_op)

    for epoch_step in range(NUM_EPOCHS):
        batch_step = 0
        batch_generator = utils.get_batch(BATCH_SIZE, train_source_int, train_target_int,
                                          source_vocab_char_to_int['<PAD>'], target_vocab_char_to_int['<PAD>'])

        for batch_pad_source_int, batch_pad_target_int, target_lengths, source_lengths in batch_generator:
            # print(batch_pad_source_int[0])
            # print(batch_pad_target_int[0])
            feed_dict_epoch = {batch_source_input_ids: batch_pad_source_int,
                               batch_target_input_ids: batch_pad_target_int,
                               target_sequences_lengths: target_lengths}
            batch_train_loss, _ = sess.run([train_model.loss, train_model.train_op],
                                           feed_dict=feed_dict_epoch)
            batch_step += 1

            if batch_step % 76 == 0:
                summary = sess.run(merged_summary, feed_dict_epoch)
                file_writer.add_summary(summary, epoch_step)

        feed_dict_test = {batch_source_input_ids: test_pad_source,
                          batch_target_input_ids: test_pad_target,
                          target_sequences_lengths: test_target_lengths}
        test_loss = sess.run(test_model.loss, feed_dict=feed_dict_test)
        summary = sess.run(merged_summary, feed_dict_test)
        file_writer.add_summary(summary, epoch_step)

        print('epoch  ' + str(epoch_step) + '/' + str(NUM_EPOCHS) + '  ' + 'training_loss:  '
              + str(batch_train_loss) + '  ' + 'test_loss:  ' + str(test_loss))

    saver.save(sess, checkpoint_dir + 'seq2seq.ckpt')
    print('Model trained and saved!')
