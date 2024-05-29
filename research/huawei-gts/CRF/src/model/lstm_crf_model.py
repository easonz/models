import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindnlp.transformers import MSBertModel
from mindspore.common.initializer import initializer, Uniform


def post_decode(score, history, seq_length):
    # 使用Score和History计算最佳预测序列
    batch_size = seq_length.shape[0]
    seq_ends = seq_length - 1
    # shape: (batch_size,)
    best_tags_list = []

    # 依次对一个Batch中每个样例进行解码
    for idx in range(batch_size):
        # 查找使最后一个Token对应的预测概率最大的标签，
        # 并将其添加至最佳预测序列存储的列表中
        best_last_tag = score[idx].argmax(axis=0)
        best_tags = [int(best_last_tag.asnumpy())]

        # 重复查找每个Token对应的预测概率最大的标签，加入列表
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(int(best_last_tag.asnumpy()))

        # 将逆序求解的序列标签重置为正序
        best_tags.reverse()
        best_tags_list.append(best_tags)
    return best_tags_list


def sequence_mask(seq_length, max_length, batch_first=False):
    """根据序列实际长度和最大长度生成mask矩阵"""
    range_vector = mnp.arange(0, max_length, 1, seq_length.dtype)
    result = range_vector < seq_length.view(seq_length.shape + (1,))
    if batch_first:
        return result.astype(ms.int64)
    return result.astype(ms.int64).swapaxes(0, 1)


# 定义Dense，基于2.2版本的ms需要使用自定义的Dense
class Dense(ms.nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 has_bias=True,
                 dtype=ms.common.dtype.float32):
        super().__init__()
        self.in_channels = ms._checkparam.check_positive_int(
            in_channels, "in_channels", self.cls_name)
        self.out_channels = ms._checkparam.check_positive_int(
            out_channels, "out_channels", self.cls_name)
        self.has_bias = ms._checkparam.check_bool(
            has_bias, "has_bias", self.cls_name)

        self.weight = ms.Parameter(ms.common.initializer.initializer(
            ms.common.initializer.HeUniform(math.sqrt(5)), [out_channels, in_channels], dtype=dtype), name="weight")

        self.bias = None
        if self.has_bias:
            fan_in, _ = ms.common.initializer._calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias = ms.Parameter(ms.common.initializer.initializer(
                ms.common.initializer.Uniform(bound), [out_channels], dtype=dtype), name="bias")

    def construct(self, x):
        x_shape = x.shape
        if len(x_shape) != 2:
            x = x.reshape(-1, x.shape[-1])
        x = mnp.matmul(x, self.weight.T)
        if self.has_bias:
            x = ops.add(x, self.bias)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (x.shape[-1],)
            x = x.reshape(out_shape)
        return x

    def extend_repr(self):
        s = f'input_channels={self.in_channels}, output_channels={self.out_channels}'
        if self.has_bias:
            s += f', has_bias={self.has_bias}'
        return s


# 用于替换ms的Dense为自定义的，解决算子转换问题
ms.nn.Dense = Dense


# 定义模型Bert-BiLSTM-CRF
class BERT_LSTM_CRF(nn.Cell):
    def __init__(self, num_tags, bert_model_path='bert-base-cased', dropout=0.0):
        super(BERT_LSTM_CRF, self).__init__()
        # 第一层，初始化Bert
        self.bert = MSBertModel.from_pretrained(bert_model_path)

        hidden_dim = self.bert.config.hidden_size

        # 第二层，初始化LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True,
                            batch_first=True, num_layers=2, dropout=dropout)

        # 第三层，初始化隐藏层Dense或者Linear
        self.hidden2tag = Dense(hidden_dim, num_tags)

        # 第四层，初始化CRF
        self.crf = CRF(num_tags, batch_first=True)

    def construct(self, inputs, seq_length, tags=None):
        # 第一层Bert
        attention_mask = (inputs > ms.Tensor(0))
        bert_out = self.bert(input_ids=inputs, attention_mask=attention_mask)

        # 第二层LSTM
        lstm_out, _ = self.lstm(bert_out[0], seq_length=seq_length)

        # 第三层Dense(或者Linear)
        hidden_out = self.hidden2tag(lstm_out)

        # 第四层CRF
        crf_out = self.crf(hidden_out, tags, seq_length)
        return crf_out


# 定义模型BiLSTM-CRF
class BiLSTM_CRF(nn.Cell):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.hidden2tag = Dense(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def construct(self, inputs, seq_length, tags=None):
        embeds = self.embedding(inputs)
        outputs, _ = self.lstm(embeds, seq_length=seq_length)
        feats = self.hidden2tag(outputs)
        crf_outs = self.crf(feats, tags, seq_length)
        return crf_outs


# 定义模型CRF
class CRF(nn.Cell):
    def __init__(self, num_tags: int, batch_first: bool = False, reduction: str = 'mean') -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.reduction = reduction
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags,)), name='start_transitions')
        self.end_transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags,)), name='end_transitions')
        self.transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags, num_tags)), name='transitions')

    def construct(self, emissions, tags=None, seq_length=None, reduction='mean'):
        if tags is None:
            return self.decode(emissions, seq_length)
        return self.forward(emissions, tags, seq_length, reduction=reduction)

    def forward(self, emissions, tags=None, seq_length=None, mask=None, reduction='mean'):
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')

        if self.batch_first:
            batch_size, max_length = tags.shape
            emissions = emissions.swapaxes(0, 1)
            tags = tags.swapaxes(0, 1)
        else:
            max_length, batch_size = tags.shape

        if seq_length is None:
            seq_length = mnp.full((batch_size,), max_length, ms.int64)
        if mask is None:
            mask = sequence_mask(seq_length, max_length)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, seq_length - 1, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = denominator - numerator
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.astype(emissions.dtype).sum()

    def decode(self, emissions, seq_length=None, mask=None):
        if self.batch_first:
            batch_size, max_length = emissions.shape[:2]
            emissions = emissions.swapaxes(0, 1)
        else:
            max_length, batch_size = emissions.shape[:2]

        if seq_length is None:
            seq_length = mnp.full((batch_size,), max_length, ms.int64)

        if mask is None:
            mask = sequence_mask(seq_length, max_length)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, seq_ends, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)

        seq_length, batch_size = tags.shape
        mask = mask.astype(emissions.dtype)

        # 将score设置为初始转移概率
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        # score += 第一次发射概率G
        score += emissions[0, mnp.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # 标签由i-1转移至i的转移概率（当mask == 1时有效）
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # 预测tags[i]的发射概率（当mask == 1时有效）
            # shape: (batch_size,)
            score += emissions[i, mnp.arange(batch_size), tags[i]] * mask[i]

        # 结束转移
        # shape: (batch_size,)
        last_tags = tags[seq_ends, mnp.arange(batch_size)]
        # score += 结束转移概率
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length = emissions.shape[0]

        # 将score设置为初始转移概率，并加上第一次发射概率
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # 扩展score的维度用于总score的计算
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.expand_dims(2)

            # 扩展emission的维度用于总score的计算
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].expand_dims(1)

            # 根据公式(7)，计算score_i
            # 此时broadcast_score是由第0个到当前Token所有可能路径
            # 对应score的log_sum_exp
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # 对score_i做log_sum_exp运算，用于下一个Token的score计算
            # shape: (batch_size, num_tags)
            next_score = ops.logsumexp(next_score, axis=1)

            # 当mask == 1时，score才会变化
            # shape: (batch_size, num_tags)
            score = mnp.where(mask[i].expand_dims(1), next_score, score)

        # 最后加结束转移概率
        # shape: (batch_size, num_tags)
        score += self.end_transitions
        # 对所有可能的路径得分求log_sum_exp
        # shape: (batch_size,)
        return ops.logsumexp(score, axis=1)

    def _viterbi_decode(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length = mask.shape[0]

        score = self.start_transitions + emissions[0]
        history = ()

        for i in range(1, seq_length):
            broadcast_score = score.expand_dims(2)
            broadcast_emission = emissions[i].expand_dims(1)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # 求当前Token对应score取值最大的标签，并保存
            indices = next_score.argmax(axis=1)
            history += (indices,)

            next_score = next_score.max(axis=1)
            score = mnp.where(mask[i].expand_dims(1), next_score, score)

        score += self.end_transitions
        return score, history
