import torch
import re
import argparse


def weight_convert(torch_model_path, ms_ckpt_path):
    torch_model = torch.load(torch_model_path, map_location="cpu")
    import mindspore as ms
    print('here')
    ms_param = []
    pattern = r'conv\d+'
    i = 0
    for key, value in torch_model["state_dict"].items():
        match = re.search(pattern, key)
        if ".0.downsample.1.weight" in key:
            key = key.replace("weight", "gamma")
        if ".0.downsample.1.bias" in key:
            key = key.replace("bias", "beta")
        if ".gn.weight" in key:
            key = key.replace("weight", "gamma")
        if ".gn.bias" in key:
            key = key.replace("bias", "beta")
        if "bn" in key and "weight" in key:
            key = key.replace("weight", "gamma")
        if "bn" in key and "bias" in key:
            key = key.replace("bias", "beta")
        if "running_mean" in key:
            key = key.replace("running_mean", "moving_mean")
        if "running_var" in key:
            key = key.replace("running_var", "moving_variance")
        if ".solo_cate." in key:
            key = key.replace(".solo_cate.",".solo_cate.conv.")
        if ".solo_kernel." in key:
            key = key.replace(".solo_kernel.",".solo_kernel.conv.")
        if match and 'mask_feat_head' in key :
            conv_str = match.group()
            key = key.replace(conv_str, conv_str[4:])
        if '.2.1.' in key:
            key = key.replace(".2.1.",".2.2.")
        elif '.3.1.' in key:
            key = key.replace(".3.1.",".3.2.")
        elif '.3.2.' in key:
            key = key.replace(".3.2.",".3.4.")
        ms_param.append({"name": key, "data": ms.Tensor(value.numpy())})
        i += 1
        print('Converting weights:', i)
    ms_param.append
    ms.save_checkpoint(ms_param, ms_ckpt_path)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('torch_weight', help='checkpoint file')
    parser.add_argument('mindspore_weight', help='coco data root')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    torch_model_path = args.torch_weight
    ms_ckpt_path = args.mindspore_weight

    print('Converting weights...')

    weight_convert(torch_model_path, ms_ckpt_path)

    print('Convertion done')

if __name__ == "__main__":
    main()