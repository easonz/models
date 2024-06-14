import sys
import argparse
import logging
import pickle
import numpy as np
import mindspore_lite as mslite



def parse_args():
    parser = argparse.ArgumentParser(description='test by mindir')
    parser.add_argument('--config', help='test config file path', default='./configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py')
    parser.add_argument('--mindir', help='mindir', required=True)
    parser.add_argument('--dataroot', help='coco data root', default='./demo.jpg')
    parser.add_argument('--out', help='output result file', default='./demo_mslite_result.jpg')
    args = parser.parse_args()
    
    return args


def build_model_by_lite(args):
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 2
    context.cpu.thread_num = 1
    context.cpu.thread_affinity_mode=2
    
    model = mslite.Model()
    logging.info(f"load model from {args.mindir} start")
    model.build_from_file(args.mindir, mslite.ModelType.MINDIR, context, config_path="./lite_config.ini")
    logging.info(f"load model from {args.mindir} success")
    
    return model



def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)d] [%(thread)d] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    args = parse_args()
    
    model = build_model_by_lite(args)
    
    with open("input_img.pkl", "rb") as fd:
        input_img_np = pickle.load(fd)
    
    inputs = model.get_inputs()
    logging.info(f"model input num:{len(inputs)}, {input_img_np.shape}")
    inputs[0].set_data_from_numpy(input_img_np)

    logging.info(f"model predict start")
    outputs = model.predict(inputs)
    logging.info(f"model predict success, outputs.len:{len(outputs)}")
    
    output_numpys = []
    for i in range(len(outputs)):
        output_data = outputs[i].get_data_to_numpy()
        output_numpys.append(output_data)
        logging.info(f"outputs[{i}] = {output_data.shape} {output_data}")
        rescale = False
    # breakpoint()
    # seg_result = solov2.bbox_head.get_seg(cate_preds= cate_preds, kernel_preds=kernel_preds, seg_pred=seg_pred, img_metas=input_img_meta, cfg=solov2.test_cfg, rescale=rescale)
    # logging.info(f"seg_result:{seg_result}")
    
    # show_result_ins(args.dataroot, seg_result, solov2.CLASSES, score_thr=0.25, out_file=args.out)
  
if __name__ == "__main__":
    main()