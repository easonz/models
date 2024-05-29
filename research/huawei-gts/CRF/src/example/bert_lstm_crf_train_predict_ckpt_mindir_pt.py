import os
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import mindspore as ms

import torch
torch.set_printoptions(precision=8)
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from transformers import BertModel
from utils.metrics import get_metric
from utils.config import config
from utils.dataset import read_data, get_dict, parse_dataset, LABEL_MAP, get_entity, GetDatasetGenerator


# 定义PT对于的Bert模型
class BERT_LSTM_CRF(nn.Module):
    def __init__(self, num_tags, bert_model_path='bert-base-cased', dropout=0.0):
        super(BERT_LSTM_CRF, self).__init__()
        # 第一层Bert
        self.bert = BertModel.from_pretrained(bert_model_path)
        hidden_dim = self.bert.config.hidden_size
        
        # 第二层LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True,
                            batch_first=True, num_layers=2, dropout=dropout)
        
        # 第三层Dense(或者Linear)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # 第四层CRF
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, mask, labels=None):
        # 第一层Bert
        attention_mask = input_ids > torch.tensor(0)
        bert_out = self.bert(input_ids, attention_mask=attention_mask)
        
        # 第二层LSTM
        lstm_out, _ = self.lstm(bert_out.last_hidden_state)
        
        # 第三层Dense(或者Linear)
        hidden_out = self.hidden2tag(lstm_out)
        
        # 第四层CRF
        if labels is not None:
            loss = -self.crf(hidden_out, labels, mask, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(hidden_out, mask)
            return prediction


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
    ms.dataset.config.set_seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    # 如果使用GPU，还需要设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sequence_mask(seq_length, max_length):
    # seq_length 是一个一维的 PyTorch Tensor，包含每个序列的实际长度
    # max_length 是一个整数，表示序列的最大长度
    # batch_first 是一个布尔值，指示是否将批次维度放在序列维度之前0

    # 使用 PyTorch 生成从 0 到 max_length - 1 的范围向量
    input = torch.tensor(seq_length)
    range_vector = torch.arange(max_length, dtype=torch.int64)
    res = range_vector < input.view(seq_length.shape + (1,))
    return res


if __name__ == "__main__":
    # Step1： 定义初始化参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = config.learning_rate
    Max_Len = config.vocab_max_length
    epochs = config.num_epochs
    dropout = 0.0
    seed_everything(42)


    # Step2: 初始化模型与优化器
    model = BERT_LSTM_CRF(bert_model_path='./example/tools/',
                          num_tags=len(LABEL_MAP), dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Step3: 训练
    train_data = read_data('../conll2003/train.txt')
    char_number, id_indexs = get_dict(train_data[0])
    train_dataset = parse_dataset(train_data, id_indexs)
    print(f'========[OK][LoadTrainDataSet]========')

    now = datetime.now()
    model.train(True)  # 设置模型为训练模式
    total = train_dataset.get_dataset_size()
    for epoch in range(epochs):
        for batch, (token_ids, seq_len, labels) in enumerate(train_dataset.create_tuple_iterator()):
            # 转换训练数据
            input_ids = torch.from_numpy(token_ids.asnumpy())
            mask = sequence_mask(seq_len.numpy(), Max_Len)
            labels = torch.from_numpy(labels.asnumpy())

            start = datetime.now()
            optimizer.zero_grad()  # 清空之前的梯度
            loss = model(input_ids, mask, labels=labels)  # 前向传播

            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            print(f'========[PT][loss]epoch: {epoch+1}/{epochs}, batch: {batch+1}/{total}, loss: {loss}, cost: {(datetime.now() - start)}')
    print(f'========[COST]========lr: {learning_rate}, tCost: {(datetime.now() - now)}')
    print(f'========[OK][Train]========')


    # Step4: 推理
    test_data = read_data('../conll2003/test.txt')
    test_dataset = parse_dataset(test_data, id_indexs)

    decodes = []
    model.train(False)
    with tqdm(total=test_dataset.get_dataset_size()) as t:
        for batch, (token_ids, seq_len, labels) in enumerate(test_dataset.create_tuple_iterator()):
            # 转换训练数据
            input_ids = torch.from_numpy(token_ids.asnumpy())
            mask = sequence_mask(seq_len.numpy(), Max_Len)
            best_tags_list = model.forward(inputs_ids.to, mask)
            decodes.extend(list(best_tags_list))
            t.update(1)

    pred = [get_entity(x) for x in decodes]
    get_metric(pred, GetDatasetGenerator(test_data, id_indexs))
    print(f'========[OK][Predict]========')
