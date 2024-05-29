import numpy as np
from tqdm import tqdm
from datetime import datetime

import mindspore as ms
from utils.config import config
# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, device_id=config.device_id)

import mindspore.nn as nn
from mindspore import Tensor
from model.lstm_crf_model import BiLSTM_CRF, post_decode
from utils.dataset import read_data,  get_dict, get_entity, seed_everything, LABEL_MAP, parse_dataset, GetDatasetGenerator


if __name__ == '__main__':
    # Step1： 定义初始化参数
    embedding_dim = config.embedding_dim
    hidden_dim = config.hidden_dim
    Max_Len = config.vocab_max_length
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.num_epochs
    seed_everything(42)


    # Step2: 读取数据集
    train_dataset = read_data('../conll2003/train.txt')
    char_number, id_indexs = get_dict(train_dataset[0])
    train_dataset_batch = parse_dataset(train_dataset, id_indexs)


    # Step3: 初始化模型与优化器
    model = BiLSTM_CRF(vocab_size=len(id_indexs), embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                       num_tags=len(LABEL_MAP))
    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    grad_fn = ms.value_and_grad(model, None, optimizer.parameters)

    def train_step(token_ids, seq_len, labels):
        loss, grads = grad_fn(token_ids, seq_len, labels)
        optimizer(grads)
        return loss
    print(f'========[OK][ModelInit]========')


    # Step4: 训练
    tloss = []
    for epoch in range(epochs):
        model.set_train()
        with tqdm(total=train_dataset_batch.get_dataset_size()) as t:
            for batch, (token_ids, seq_len, labels) in enumerate(train_dataset_batch.create_tuple_iterator()):
                loss = train_step(token_ids, seq_len, labels)
                tloss.append(loss.asnumpy())
                t.set_postfix(loss=np.array(tloss).mean())
                t.set_postfix_str(f'epoch: {epoch}/{epochs}')
                t.update(1)
    print(f'========[OK][Train]========')


    # Step5: 推理
    test_data = read_data('../conll2003/test.txt')
    test_dataset = parse_dataset(test_data, id_indexs)

    decodes = []
    model.set_train(False)
    with tqdm(total=test_dataset.get_dataset_size()) as t:
        for batch, (token_ids, seq_len, labels) in enumerate(test_dataset.create_tuple_iterator()):
            score, history = model(token_ids, seq_len)
            best_tag = post_decode(score, history, seq_len)
            decode = [[y for y in x] for x in best_tag]
            decodes.extend(list(decode))
            t.update(1)

    pred = [get_entity(x) for x in decodes]
    get_metric(pred, GetDatasetGenerator(test_data, id_indexs))
    print(f'========[OK][Predict]========')


    # Step6: 导出CKPT
    mmdd_hhmm = datetime.now().strftime("%m%d-%H%M")
    file_name = f'lstm_crf-{mmdd_hhmm}'
    ms.save_checkpoint(model, ckpt_file_name=f'{file_name}.ckpt')
    print(f'========[OK][CKPT]========Create CKPT SUCCEEDED, file: {file_name}')


    # Step7: 导出MindIR
    input0 = Tensor(np.ones((batch_size, Max_Len)).astype(np.int32))
    input1 = Tensor(np.ones(batch_size).astype(np.int32))
    ms.export(model, input0, input1, file_name=file_name, file_format='MINDIR')
    print(f'========[OK][MindIR]========Create MINDIR SUCCEEDED, file: {file_name}')