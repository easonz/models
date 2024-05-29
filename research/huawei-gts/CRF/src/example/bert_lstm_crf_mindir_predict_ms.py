from datetime import datetime
from tqdm import tqdm

import mindspore as ms
from utils.config import config
# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, device_id=config.device_id)
import mindspore.nn as nn

from model.lstm_crf_model import post_decode
from utils.dataset import read_data, GetDatasetGenerator, get_dict, get_entity, parse_dataset
from utils.metrics import get_metric


if __name__ == '__main__':
    # Step1： 定义初始化参数
    batch_size = config.batch_size

    # Step2: 通过mindir加载模型，这里可以传入路径
    file_name = 'bert_lstm_crf.mindir'
    graph = ms.load(file_name)
    model = nn.GraphCell(graph)
    print(model)

    # Step3: 读取数据集
    train_dataset = read_data('../conll2003/train.txt')
    test_dataset = read_data('../conll2003/test.txt')
    char_number, id_indexs = get_dict(train_dataset[0])
    test_dataset_batch = parse_dataset(test_dataset, id_indexs)

    # Step4: 进行预测
    now = datetime.now()
    decodes = []
    model.set_train(False)
    with tqdm(total=test_dataset_batch.get_dataset_size()) as t:
        for batch, (token_ids, seq_len, labels) in enumerate(test_dataset_batch.create_tuple_iterator()):
            score, history = model(token_ids, seq_len)
            best_tag = post_decode(score, history, seq_len)
            decode = [[y for y in x] for x in best_tag]
            decodes.extend(list(decode))
            t.update(1)
    print(f'========[COST]======== Total time: {(datetime.now() - now)}')

    pred = [get_entity(x) for x in decodes]
    get_metric(pred, GetDatasetGenerator(test_dataset, id_indexs))
