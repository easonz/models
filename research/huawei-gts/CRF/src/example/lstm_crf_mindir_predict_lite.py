import numpy as np
from tqdm import tqdm
from datetime import datetime

import mindspore_lite as mslite

from model.lstm_crf_model import post_decode
from utils.dataset import read_data, GetDatasetGenerator, get_dict, parse_dataset, get_entity
from utils.metrics import get_metric

if __name__ == '__main__':
    # Step1: 定义mslite运行参数
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0
    context.cpu.thread_num = 1
    context.cpu.thread_affinity_mode = 2
    
    # Step2: 初始化模型
    MODEL_PATH = 'lstm_crf.mindir'
    model = mslite.Model()
    model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context)
    print(f'========[MODEL][OK]====== build_from_file: {model}')
    
    # Step3: 读取数据集
    train_dataset = read_data('../conll2003/train.txt')
    test_dataset = read_data('../conll2003/test.txt')
    char_number, id_indexs = get_dict(train_dataset[0])
    test_dataset_batch = parse_dataset(test_dataset, id_indexs)

    # Step5: 进行预测
    now = datetime.now()
    decodes = []
    with tqdm(total=test_dataset_batch.get_dataset_size()) as t:
        for batch, (token_ids, seq_length, labels) in enumerate(test_dataset_batch.create_tuple_iterator()):
            inputs = model.get_inputs()
            inputs[0].set_data_from_numpy(
                token_ids.asnumpy().astype(dtype=np.int32))
            inputs[1].set_data_from_numpy(
                seq_length.asnumpy().astype(dtype=np.int32))
            outputs = model.predict(inputs)
            score, history = outputs[0], outputs[1:]
            score = score.get_data_to_numpy()
            history = list(map(lambda x: x.get_data_to_numpy(), history))
            best_tags = post_decode(score, history, seq_length.asnumpy())
            decode = [[y for y in x] for x in best_tags]
            decodes.extend(list(decode))
    print(f'========[COST]======== Total time: {(datetime.now() - now)}')

    pred = [get_entity(x) for x in decodes]
    get_metric(pred, GetDatasetGenerator(test_dataset, id_indexs))
