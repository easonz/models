import numpy as np
from tqdm import tqdm
import torch
import mindspore_lite as mslite

from utils.config import config
from utils.dataset import get_entity, read_data, get_dict, GetDatasetGenerator
from utils.metrics import get_metric


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
        best_tags = [int(best_last_tag)]

        # 重复查找每个Token对应的预测概率最大的标签，加入列表
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(int(best_last_tag))

        # 将逆序求解的序列标签重置为正序
        best_tags.reverse()
        best_tags_list.append(best_tags)
    return best_tags_list


def pad(batch):
    '''Pads to the longest sample'''
    ids = None
    lables = None
    seq_lens = None
    for sample in batch:
        id,seq_len,lab = sample
        if ids is None:
            ids = torch.tensor(id).unsqueeze(0)
        else:
            ids  = torch.cat([ids,torch.tensor(id).unsqueeze(0)],0)
        if lables is None:
            lables = torch.tensor(lab).unsqueeze(0)
        else:
            lables = torch.cat([lables,torch.tensor(lab).unsqueeze(0)],0)
        if seq_lens is None:
            seq_lens = torch.tensor(seq_len).unsqueeze(0)
        else:
            seq_lens = torch.cat([seq_lens,torch.tensor(seq_len).unsqueeze(0)],0)
    
    return ids,lables,seq_lens


# 执行测试用例前，确保参照ReadME中，将conll2003数据集下载并放入指定位置
if __name__ == '__main__':
    # Step1: 定义mindir运行的上下文
    context = mslite.Context()
    # 设置运行环境这里设置在 ascend\cpu\gpu进行推理
    context.target = ["ascend"]
    # 这里可以设置使用哪个NPU进行推理，一般0是第一块NPU
    context.ascend.device_id = 0
    context.cpu.thread_num = 1
    context.cpu.thread_affinity_mode = 2
    batch_size = config.batch_size
    

    # Step2: 初始化模型
    model = mslite.Model()
    file_name = "bert_lstm_crf.mindir"
    model.build_from_file(file_name, mslite.ModelType.MINDIR, context)
    print(model)


    # Step3: 读取词表，注意这里的train数据需要与ckpt训练的数据一致
    train = read_data('../conll2003/train.txt')
    char_number_dict, id_indexs = get_dict(train[0])


    # Step4: 使用test数据集进行预测
    test = read_data('../conll2003/test.txt')
    test_dataset = GetDatasetGenerator(test, id_indexs)
    from torch.utils.data import DataLoader
    train_iter = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad)


    # Step5: 进行推理
    decodes=[]
    with tqdm(total=len(test_dataset)//batch_size) as t:
        for batch in train_iter:
            token_ids, lables, seq_len = batch
            
            if seq_len.shape[0] != batch_size:
                break
            
            inputs = model.get_inputs()
            inputs[0] = mslite.Tensor(token_ids.numpy().astype(dtype=np.int32))
            inputs[1] = mslite.Tensor(seq_len.numpy().astype(dtype=np.int32))

            outputs = model.predict(inputs)
            score, history = outputs[0], outputs[1:]
            score = score.get_data_to_numpy()
            history = list(map(lambda x: x.get_data_to_numpy(), history))
            best_tags = post_decode(score, history, seq_len.numpy())
            decode = [[y for y in x] for x in best_tags]
            decodes.extend(list(decode))
            t.update(1)
    
    pred = [get_entity(x) for x in decodes]
    get_metric(pred, test_dataset)
