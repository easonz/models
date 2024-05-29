import torch
from mindspore import Parameter, Tensor, load_checkpoint, save_checkpoint


# 加载checkpoint文件
ckpt_file = "/home/y30007104/1.code/mindnlp-001/models-CRF0513/research/huawei-gts/CRF/src/bert_lstm_crf-0525-1901-111.ckpt"
pt_file = "/home/y30007104/1.code/g00835429_test/torch.pt"
save_file = "/home/y30007104/1.code/mindnlp-001/models-CRF0513/research/huawei-gts/CRF/src/gyw.ckpt"

param_dict = load_checkpoint(ckpt_file)
# 可以通过遍历这个dict，查看key和value

torch_dict = torch.load(pt_file, map_location='cpu')

for key, value in param_dict.items():
    print(key)
    print(torch_dict[key].shape)

    param_dict[key] = Parameter(Tensor(torch_dict[key].cpu().numpy()))
    # key 为string类型

    # value为parameter类型，使用data.asnumpy()方法可以查看其数值
    # print(value.data.asnumpy())

# 拿到param_dict后，就可以对其进行基本的增删操作，以便后续使用

# # 1.删除名称为"conv1.weight"的元素
# del param_dict["conv1.weight"]
# # 2.添加名称为"conv2.weight"的元素，设置它的值为0
# param_dict["conv2.weight"] = Parameter(Tensor([0]))
# # 3.修改名称为"conv1.bias"的值为1
# param_dict["fc1.bias"] = Parameter(Tensor([1]))

# 把修改后的param_dict重新存储成checkpoint文件
save_list = []
# 遍历修改后的dict，把它转化成MindSpore支持的存储格式，存储成checkpoint文件
for key, value in param_dict.items():
    save_list.append({"name": key, "data": value.data})
save_checkpoint(save_list, save_file)
