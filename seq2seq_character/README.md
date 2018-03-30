# seq2seq_character
## 概括
该项目实现了一个基本的Seq2Seq模型，接收一串英文字母（不含标点、空格），然后安装字母表顺序输出。包括以下部分：
- seq2seq_model.py: 实现seq2seq模型

- train_test.py: 用于训练测试

- predict.py: 用于预测

- utils.py: 读取数据、处理数据等接口

## 主要类、函数介绍：
#### Encoder类:
编码器，将输入信息编码为final_states，作为解码器的初始状态信息。

参数：
- *source_input_ids: 输入信息，shape: (batch_size, max_sequence_length) ，max_sequence_length指每批输入的最长序列长度*
- *num_units: 每个cell的隐藏神经元数目*
- *num_layers: 编码器层数*
- *source_vocab_size: 输入词汇表大小*
- *train_test_predict: 字符串类型，‘train’代表训练，‘test’代表测试，‘predict’代表预测*
- *keep_prob: 用于druopout层的概率参数*

#### Decoder类:
解码器，将编码器的final_states解码为输出信息。

参数：
- *train_test_predict: 字符串类型，‘train’代表训练，‘test’代表测试，‘predict’代表预测*
- *processed_target_input_ids: 预处理(去除每个单词最后一个字符，并在词首加字符‘<GO>')后的解码器输入，shape: (batch_size, max_sequence_length)，max_sequence_length指每批输入的最长序列长度*
- *initial_state: 初始状态信息，Encoder返回的final_states*
- *sequences_lengths: 解码器输入词汇的真实长度(没有padding)，shape: [batch_size]*
- *target_vocab_size: 目标词汇表大小*
- *num_units: 每个cell的隐藏神经元数目*
- *num_layers: 解码器层数*
- *max_sequences_length: 指每批输入的最长序列长度*
- *start_tokens: 用于预测过程，shape[batch_size]*
- *end_token: int类型，用于预测过程*

#### Seq2seqModel类:
将编码器解码器整合为一个seq2seq模型(包含优化、训练信息)。

参数：
- *source_input_ids: 编码器输入信息，shape: (batch_size, max_sequence_length)，max_sequence_length指编码器每批输入的最长序列长度*
- *target_input_ids: 解码器输入信息，shape: (batch_size, max_sequence_length)，max_sequence_length指解码器每批输入的最长序列长度*
- *num_units: 每个cell的隐藏神经元数目*
- *num_layers: 解码器层数*
- *source_vocab_size: 输入词汇表大小*
- *target_vocab_size: 目标词汇表大小*
- *target_sequences_lengths: 解码器输入词汇的真实长度(没有padding)，shape: [batch_size]*
- *train_test_predict: 字符串类型，‘train’代表训练，‘test’代表测试，‘predict’代表预测*
- *grad_clip_norm: 用于梯度裁剪，防止梯度爆炸*
- *learning_rate: 学习率*
- *target_vocab_char_to_int: dict类型，用以将目标字母映射成整型数字*
- *batch_size: 每批大小*
- *max_sequences_length: 指解码器每批输入的最长序列长度*
- *keep_prob: 用于druopout层的概率参数*
- *start_tokens: 用于预测过程，shape[batch_size]*
- *end_token: int类型，用于预测过程*
