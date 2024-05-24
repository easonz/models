# Solov2

## Solov2概述

Solov2是一种box-free的实列分割算法，Solov2在第一版solo的基础上针对mask的检测效果和运行效率做出了改进。

本仓库将原始Pytorch实现，迁移到Mindspore框架（**当前为初始版本，持续更新中**...）



参考论文：SOLO v2: Dynamic, Faster and Stronger



## 环境要求

本工程测试验证配套关系

| 环境 | 驱动&固件            | CANN    | mindspore |
| ---- | -------------------- | ------- | --------- |
| 910B | 23.0.3 & 7.1.0.5.220 | 8.0.RC1 | 2.3.0rc1  |



## 预训练模型

本工程分析验证了以下模型，可以通过HuggingFace超链接下载对应的权重文件

| Model              | Multi-scale training | Testing time / im | AP (minival) | Link                                                         |
| ------------------ | -------------------- | ----------------- | ------------ | ------------------------------------------------------------ |
| SOLOv2_R101_DCN_3x | 待补充               | 待补充            | 待补充       | [Download](https://huggingface.co/xinlongwang/SOLO/resolve/main/SOLOv2_R101_DCN_3x.pth?download=true ) |





## 快速入门


### 权重转换
运行权重转换脚本实现权重转换
python weight_convert.py 'torch权重地址' '输出权重地址'
```shell
# for example
python weight_convert.py './pretrained_weights/SOLOv2_R101_DCN_3x.pth' './pretrained_weights/SOLOv2_R101_DCN_3x.ckpt'
```



### 运行推理

运行单图片推理：
python test.py '配置文件地址' '权重文件地址' --dataroot '推理原图地址(默认为根路径下的demo.jpg)' --out '输出图片地址(默认为demo_ms.jpg)'

```shell
# for example:
python test.py ./configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py ./SOLOv2_R101_DCN_3x.ckpt --dataroot ./demo.jpg --out ./demo_ms_out.jpg
```

运行多图片推理：
python run.py '配置文件地址' '权重文件地址' '待处理图片所在文件夹地址' --out '输出图片地址(默认为./out_put)'
```shell
# for example:
python run.py configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py ./SOLOv2_R101_DCN_3x.ckpt ./coco/test --out ./outfile
```



### 运行评估

运行valuation.py脚本
python run.py '配置文件地址' '权重文件地址' '数据集所在文件夹地址' --show  --out '输出文件地址' --eval 'eval类型 有['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints']'
```shell
# for example:
python valuation.py configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py ./SOLOv2_R101_DCN_3x-fpn.ckpt /data/coco --show --out results_solo.pkl --eval segm
```




### 运行训练
#### 数据集

TrashCan 是一个水下垃圾的实例分割数据集，由 7,212 张标注图像组成，记录了各种水下垃圾、无人潜水器和海底动植物群的活动。该数据集的标注采用了实例分割标注的格式，其图像来源于 J-EDI 数据集。

数据集下载地址：https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7

训练数据：6065张

验证数据：1147张

#### 参数配置

TrashCan数据集与CoCo数据集格式基本一致，需要修改config的solov2_r101_dcn_fpn_8gpu_3x.py文件中的配置

```
# model settings
model = dict(
    ...
    bbox_head=dict(
        type='SOLOv2Head',
        num_classes=23,            # 修改类别数量 = 类别 + 1 （还有一类为背景） 
        in_channels=256,
        stacked_convs=4,
        use_dcn_in_tower=True,
        type_dcn='DCNv2',
        seg_feat_channels=512,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
        ...
        ),
    )
...
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'path/to/trashcan/instance_version/'   # 修改数据集地址，此处使用instance_version
...
# 修改ann_file和img_prefix为对应地址
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_train_trashcan.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_val_trashcan.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'instances_val_trashcan.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # 设置学习率
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[27, 33])

checkpoint_config = dict(interval=1)  # 设置模型保存频率
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 36                          # 设置训练轮次
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/solov2_release_r101_dcn_fpn_8gpu_3x'  # 设置模型保存地址
load_from = '/path/to/checkpoint'      # 设置预训练模型，使用coco训练好的预训练模型
resume_from = None
workflow = [('train', 1)]

```

cocodataset.py文件中，需要将原有的coco数据集80类，改成新数据集的类别

```
class CocoDataset():
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
    #            'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
    #            'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    #            'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    
    tmp_category_ids = {
    'rov':                      1,
    'plant':                    2,
    'animal_fish':              3,
    'animal_starfish':          4,
    'animal_shells':            5,
    'animal_crab':              6,
    'animal_eel':               7,
    'animal_etc':               8,
    'trash_clothing':           9,
    'trash_pipe':               10,
    'trash_bottle':             11,
    'trash_bag':                12,
    'trash_snack_wrapper':      13,
    'trash_can':                14,
    'trash_cup':                15,
    'trash_container':          16,
    'trash_unknown_instance':   17,
    'trash_branch':             18,
    'trash_wreckage':           19,
    'trash_tarp':               20,
    'trash_rope':               21,
    'trash_net':                22
    }
    CLASSES = tuple(tmp_category_ids.keys())
```



#### 单卡微调

单卡训练启动命令

```
python train.py --config configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py
```



#### 多卡微调

- 启动分布式训练，需要先运行hccl_tools.py，手动生成**RANK_TABLE_FILE**，生成命令如下：

```
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./hccl_tools.py --device_num "[0,8)"
```

RANK_TABLE_FILE 单机8卡参考样例:

```
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

- 执行分布式微调命令

```
bash run_singlenode.sh "python train.py --config configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py" /path/to/RANK_TABLE_FILE.json [0,8] 8
```

