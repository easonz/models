# Solov2

## Solov2概述

Solov2是一种box-free的实列分割算法，Solov2在第一版solo的基础上针对mask的检测效果和运行效率做出了改进。

本仓库将原始Pytorch实现，迁移到Mindspore框架（**当前为初始版本，持续更新中**...）



参考论文：SOLO v2: Dynamic, Faster and Stronger



## 环境要求

本工程测试验证配套关系

| 环境 | 驱动&固件            | CANN    | mindspore |
| ---- | -------------------- | ------- | --------- |
| 910B | 23.0.3 & 7.1.0.5.220 | 8.0.RC1 | 2.3.0rc2  |



## 预训练模型

本工程分析验证了以下模型，可以通过HuggingFace超链接下载对应的权重文件

| Model          | Multi-scale training | Testing time / im | AP (minival) | Link                                                         |
| -------------- | -------------------- | ----------------- | ------------ | ------------------------------------------------------------ |
| SOLOv2_R101_3x | 待补充               | 待补充            | 待补充       | [Download](https://huggingface.co/xinlongwang/SOLO/resolve/main/SOLOv2_R101_3x.pth?download=true) |





## 快速入门


### 权重转换
运行权重转换脚本实现权重转换
python weight_convert.py 'torch权重地址' '输出权重地址'
```shell
# for example
python weight_convert.py './pretrained_weights/SOLOv2_R101_3x.pth' './pretrained_weights/SOLOv2_R101_3x.ckpt'
```


### 运行推理

运行单图片推理(mindspore在线推理)：
python test.py '配置文件地址' '权重文件地址' --dataroot '推理原图地址(默认为根路径下的demo.jpg)' --out '输出图片地址(默认为demo_ms.jpg)'

```shell
# for example:
python test.py --config ./configs/solov2/solov2_r101_fpn_8gpu_3x.py --checkpoint ./SOLOv2_R101_3x.ckpt --dataroot ./demo.jpg --out ./demo_ms_out.jpg
#或
python test.py --config ./configs/solov2/solov2_r101_fpn_8gpu_3x.py --checkpoint ./SOLOv2_R101_3x.ckpt --dataroot ./demo.jpg --out ./demo_ms_out.jpg
```

运行多图片推理：
python run.py '配置文件地址' '权重文件地址' '待处理图片所在文件夹地址' --out '输出图片地址(默认为./out_put)'

```shell
# for example:
python run.py configs/solov2/solov2_r101_fpn_8gpu_3x.py ./SOLOv2_R101_3x.ckpt ./coco/test --out ./outfile
```

运行单图片推理(MindsporeLite离线推理)：

```shell
#在910B上导出mindir:
python export.py --config ./configs/solov2/solov2_r101_fpn_8gpu_3x.py --checkpoint ./SOLOv2_R101_3x.ckpt --out_file_name=out --out_file_format=mindir
#或
python export.py --config ./configs/solov2/solov2_r101_fpn_8gpu_3x.py --checkpoint ./SOLOv2_R101_3x.ckpt --out_file_name=out --out_file_format=mindir

#在910B或310P上执行convert.sh生成out_lite.mindir和lite推理
bash convert.sh
python test_by_mindspore_lite.py --mindir out_lite.mindir
```

### 运行评估

运行valuation.py脚本
python run.py '配置文件地址' '权重文件地址' '数据集所在文件夹地址' --show  --out '输出文件地址' --eval 'eval类型 有['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints']'
```shell
# for example:
python valuation.py configs/solov2/solov2_r101_fpn_8gpu_3x.py ./SOLOv2_R101_3x-fpn.ckpt /data/coco --show --out results_solo.pkl --eval segm
```




### 运行训练
待补充...