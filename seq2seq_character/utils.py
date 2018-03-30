import tensorflow as tf


def read_data(source_path, target_path):
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = f.read()
    with open(target_path, 'r', encoding='utf-8') as f:
        target_data = f.read()
    return source_data.split('\n'), target_data.split('\n')


def construct_character_vocab(source_data_list, target_data_list):
    """Construct the source and target vocabularies

    :param source_data_list: a list of source data, eg. returned by read_data
    :param target_data_list: a list of target data, eg. returned by read_data
    :return: source_vocab_char_to_int, source_vocab_int_to_char,
             target_vocab_char_to_int, target_vocab_int_to_char
    """

    def vocab(data):
        """
        :param data: a list
        :return: character_to_int, int_to_character
        """
        special_chars = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        # sorted_chars = special_chars + sorted(Counter([char for line in data for char in line]).keys())
        sorted_chars = special_chars + sorted(list(set([char for line in data for char in line])))
        int_to_character = {ids: ch for ids, ch in enumerate(sorted_chars)}
        character_to_int = {ch: ids for ids, ch in enumerate(sorted_chars)}

        return character_to_int, int_to_character

    source_vocab_char_to_int, source_vocab_int_to_char = vocab(source_data_list)
    target_vocab_char_to_int, target_vocab_int_to_char = vocab(target_data_list)

    return source_vocab_char_to_int, source_vocab_int_to_char, \
           target_vocab_char_to_int, target_vocab_int_to_char


def source_target_to_int(source_data_list, target_data_list,
                         source_vocab_char_to_int, target_vocab_char_to_int):
    """

    :param source_data_list: a list of source data, eg. returned by read_data
    :param target_data_list: a list of target data, eg. returned by read_data
    :param source_vocab_char_to_int: vocabulary mapping source character to integer
    :param target_vocab_char_to_int: vocabulary mapping target character to integer
    :return: source_int, target_int
    """
    source_int = [[source_vocab_char_to_int.get(char, source_vocab_char_to_int['<UNK>']) for char in line]
                  for line in source_data_list]
    target_int = [[target_vocab_char_to_int.get(char, target_vocab_char_to_int['<UNK>']) for char in line]
                  + [target_vocab_char_to_int['<EOS>']] for line in target_data_list]

    return source_int, target_int


def process_decoder_inputs(target_inputs_ids, batch_size, target_vocab_char_to_int, start_token='<GO>'):
    """Delete the last character of each example and add the start token to the front

    :param target_inputs_ids: a list, shape (batch_size, time_steps)
    :param batch_size: size of current batch
    :param target_vocab_char_to_int: a dict, returned by construct_character_vocab
    :param start_token: the start token to be added to the target inputs
    :return: processed target_inputs_ids
    """
    # delete the last character of each example
    deleted = tf.strided_slice(target_inputs_ids, [0, 0], [batch_size, -1], [1, 1])
    output = tf.concat([tf.fill([batch_size, 1], target_vocab_char_to_int['<GO>']), deleted],
                                  axis=1)
    return output


def pad_batch(batch_input, pad_int):
    """Pad the batch_input to equivalent length
    """
    max_length = max([len(line) for line in batch_input])
    return [line + [pad_int]*(max_length-len(line)) for line in batch_input]


def get_batch(batch_size, source_int, target_int, source_pad_int, target_pad_int):
    """Define a generator to yield a batch of data

    """
    times = len(source_int) // batch_size

    for i in range(times):
        start = i * batch_size
        batch_source = source_int[start: start+batch_size]
        batch_target = target_int[start: start+batch_size]

        # Record the real lengths of target_int which will be used to compute losses
        target_lengths = []
        target_lengths.extend(len(line) for line in batch_target)
        source_lengths = []
        source_lengths.extend(len(line) for line in batch_source)

        batch_pad_source_int = pad_batch(batch_source, source_pad_int)
        batch_pad_target_int = pad_batch(batch_target, target_pad_int)

        yield batch_pad_source_int, batch_pad_target_int, target_lengths, source_lengths


def get_inputs():
    batch_source_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='batch_source_input_ids')
    batch_target_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='batch_target_input_ids')
    target_sequences_lengths = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_sequences_lengths')
    max_target_sequences_length = tf.reduce_max(target_sequences_lengths)

    return batch_source_input_ids, batch_target_input_ids, target_sequences_lengths, max_target_sequences_length


if __name__ == '__main__':
    source_data_list, target_data_list = read_data('./data/letters_source.txt', './data/letters_target.txt')
    # print(source_data[: 5])
    # print(target_data[: 5])
    source_vocab_char_to_int, source_vocab_int_to_char, target_vocab_char_to_int, target_vocab_int_to_char = \
        construct_character_vocab(source_data_list, target_data_list)

    source_int, target_int = source_target_to_int(source_data_list, target_data_list,
                                                  source_vocab_char_to_int, target_vocab_char_to_int)

    print(source_int[:5])
    print(target_int[:5])

    test_source_int = source_int[: 128]
    test_target_int = target_int[: 128]
    batch_generator = get_batch(128, test_source_int, test_target_int,
                                source_vocab_char_to_int['<PAD>'], target_vocab_char_to_int['<PAD>'])
    for batch_pad_source_int, batch_pad_target_int, target_lengths, source_lengths in batch_generator:
        print(batch_pad_source_int[0])
        print(batch_pad_target_int[0])
