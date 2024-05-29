from tqdm import tqdm
from datetime import datetime

import mindspore as ms
from utils.config import config
# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, device_id=config.device_id)

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model.lstm_crf_model import BERT_LSTM_CRF, post_decode
from utils.dataset import read_data, GetDatasetGenerator, parse_dataset, seed_everything, get_dict, get_entity, LABEL_MAP
from utils.metrics import get_metric


if __name__ == '__main__':
    # Step1： 定义初始化参数
    batch_size = config.batch_size
    seed_everything(42)


    # Step2: 加载ckpt，传入文件路径与名称
    file_name = 'bert_lstm_crf.ckpt'
    param_dict = load_checkpoint(file_name)


    # Step3: 初始化模型
    model = BERT_LSTM_CRF(bert_model_path='./example/tools/',
                          num_tags=len(LABEL_MAP), dropout=0.0)

    # Step4: 将ckpt导入model
    load_param_into_net(model, param_dict)
    print(model)

    # Step5: 读取数据集
    train_dataset = read_data('../conll2003/train.txt')
    test_dataset = read_data('../conll2003/test.txt')
    char_number, id_indexs = get_dict(train_dataset[0])

    test_data = read_data('../conll2003/test.txt')
    test_dataset = parse_dataset(test_data, id_indexs)

    # Step6: 进行预测
    now = datetime.now()
    decodes = []
    model.set_train(False)
    with tqdm(total=test_dataset.get_dataset_size()) as t:
        for batch, (token_ids, seq_len, labels) in enumerate(test_dataset.create_tuple_iterator()):
            score, history = model(token_ids, seq_len)
            best_tag = post_decode(score, history, seq_len)
            decode = [[y for y in x] for x in best_tag]
            decodes.extend(list(decode))
            t.update(1)
    print(f'========[COST]======== Total time: {(datetime.now() - now)}')
    pred = [get_entity(x) for x in decodes]
    get_metric(pred, GetDatasetGenerator(test_data, id_indexs))
