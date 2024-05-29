from datetime import datetime
import mindspore as ms

from utils.config import config
ms.set_context(device_target=config.device_target, device_id=config.device_id, mode=ms.GRAPH_MODE)
import mindspore.ops as ops

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model.lstm_crf_model import BiLSTM_CRF
from utils.dataset import LABEL_MAP

# 执行测试用例前，确保参照ReadME中，将conll2003数据集下载并放入指定位置
if __name__ == '__main__':
    # Step1： 定义初始化参数
    learning_rate = config.learning_rate
    epochs = config.num_epochs
    Max_Len = config.vocab_max_length
    batch_size = config.batch_size

    # Step2: 加载ckpt，传入文件路径与名称
    file_name = 'lstm_crf.ckpt'
    param_dict = load_checkpoint(file_name)

    # Step3: 获取模型初始化参数
    embedding_shape = param_dict.get('embedding.embedding_table').shape

    # Step4: 初始化模型
    model = BiLSTM_CRF(vocab_size=embedding_shape[0], embedding_dim=embedding_shape[1], hidden_dim=embedding_shape[1],
                       num_tags=len(LABEL_MAP))

    # Step5: 将ckpt导入model
    load_param_into_net(model, param_dict)
    print(model)
    
    # Step6: 导出MindIR
    mmdd_hhmm = datetime.now().strftime("%m%d-%H%M")
    file_name = f'lstm_crf-{mmdd_hhmm}'

    ms.export(model, ops.ones((batch_size, Max_Len), ms.int32),
            ops.ones(batch_size, ms.int32),
            file_name=file_name, file_format="MINDIR")
    print(f"export mindir success : {file_name}")