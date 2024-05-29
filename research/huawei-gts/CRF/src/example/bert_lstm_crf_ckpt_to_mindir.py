import numpy as np
from datetime import datetime
import mindspore as ms
# 设置mindspore的执行目标，可以使Ascend、CPU、GPU，mode建议位图模式。注意，ms需要放到import的首行，避免context设置不生效
ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, device_id=config.device_id)

from utils.config import config

from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model.lstm_crf_model import BERT_LSTM_CRF
from utils.dataset import LABEL_MAP


if __name__ == '__main__':
    # Step1： 定义初始化参数
    Max_Len = config.vocab_max_length
    batch_size = config.batch_size
    
    # Step2: 加载ckpt，传入文件路径与名称
    file_name = 'bert_lstm_crf.ckpt'
    param_dict = load_checkpoint(file_name)
    
    
    # Step3: 初始化模型
    model = BERT_LSTM_CRF(num_tags=len(LABEL_MAP), bert_model_path='./example/tools/')
    
    # Step4: 将ckpt导入model
    param_not_load, _ = load_param_into_net(model, param_dict)
    # param_not_load是未被加载的参数列表，为空时代表所有参数均加载成功。
    print(param_not_load)

    # Step5: 导出MindIR
    mmdd_hhmm = datetime.now().strftime("%m%d-%H%M")
    file_name = f'bert_lstm_crf-{mmdd_hhmm}'

    input0 = Tensor(np.ones((batch_size, Max_Len)).astype(np.int32))
    input1 = Tensor(np.ones(batch_size).astype(np.int32))
    
    ms.export(model, input0, input1, file_name=file_name, file_format="MINDIR")
    print(f'========[OK][CKPT -> MindIR]========Create MINDIR SUCCEEDED, file: {file_name}')