import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from torchcrf import CRF
from utils.config import config
from utils.dataset import LABEL_MAP
from utils.dataset import read_data, get_dict, get_entity, parse_dataset, LABEL_MAP


class BERT_LSTM_CRF(nn.Module):
    def __init__(self, bert_model_name, num_tags, hidden_dim, lstm_layers, dropout):
        super(BERT_LSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim,
                            lstm_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        lstm_feats = self.fc(lstm_out)

        if labels is not None:
            loss = -self.crf(lstm_feats, labels, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(lstm_feats)
            return prediction


def generate_mask(batch_seq_lens, max_seq_len):
    """_summary_
    生成 CRF 所需的 mask。
    Args:
        batch_seq_lens (int): 序列的实际长度。
        max_seq_len (int): 批次中所有序列被 padding 到的最大长度。

    Returns:
        tensor: 一个布尔类型的 tensor，形状为 (batch_size, max_seq_len)。
    """

    batch_size = len(batch_seq_lens)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    for i, seq_len in enumerate(batch_seq_lens):
        mask[i, :seq_len] = 1

    return mask


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = BERT_LSTM_CRF('bert-base-uncased', len(LABEL_MAP), 256, 2, 0)
    model = BERT_LSTM_CRF(bert_model_name='./example/tools/', num_tags=len(LABEL_MAP),
                          hidden_dim=config.hidden_dim, lstm_layers=config.num_layers, dropout=0)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_data = read_data('../conll2003/train.txt')
    char_number, id_indexs = get_dict(train_data[0])
    train_dataset = parse_dataset(train_data, id_indexs)

    now = datetime.now()
    Epochs = config.num_epochs
    for epoch in range(Epochs):
        model.train()  # 设置模型为训练模式
        with tqdm(total=train_dataset.get_dataset_size()) as t:
            for batch, (token_ids, seq_len, labels) in enumerate(train_dataset.create_tuple_iterator()):
                optimizer.zero_grad()  # 清空之前的梯度

                # 转换训练数据
                imput_ids = torch.from_numpy(token_ids.asnumpy())
                mask = generate_mask(seq_len, config.vocab_max_length)
                labels = torch.from_numpy(labels.asnumpy())
                outputs = model(imput_ids, mask, labels=labels)  # 前向传播

                loss = outputs  # 假设模型返回的是损失值
                print(f'========[model][loss] loss: {loss}')
                loss.backward()  # 反向传播
                optimizer.step()  # 更新模型参数
                t.set_postfix_str(f'epoch: {epoch + 1}/{Epochs}')
                t.update(1)

    print(f'========[COST]======== Total time: {(datetime.now() - now)}')
