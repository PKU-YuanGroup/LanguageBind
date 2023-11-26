import json
import logging
import os

import numpy as np
import torch
from scipy.special import softmax
from training.distributed import is_master
from .zero_shot import zero_shot_eval


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]

def merge(eval_path, num_tasks, method='prob'):
    assert method in ['prob', 'score']
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    # logging.info("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(
                line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            if method == 'prob':
                dict_feats[name].append(softmax(data))
            else:
                dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    # logging.info("Computing final results")

    input_lst = []
    # logging.info(f"{len(dict_feats)}")
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    # pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    return final_top1 * 100, final_top5 * 100


# def evaluate_v_cls(model, data, epoch, args, tb_writer=None):
#     model.eval()
#     dataloader = data['v_cls']
#     args.output_dir = os.path.join(args.log_base_path, 'video_cls')
#     os.makedirs(args.output_dir, exist_ok=True)
#     if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
#         if is_master(args):
#             logging.info(f"Eval Epoch: {epoch}, accuracy of zero-shot classification under Kinetics-400 test videos")
#         zero_shot_eval(model, dataloader, epoch, args)
#
#     torch.distributed.barrier()
#
#     if is_master(args):
#         # logging.info("Start merging results...")
#         final_top1, final_top5 = merge(args.output_dir, args.world_size)
#         logging.info(f"\t>>>  Acc@1: {final_top1:.2f}%, Acc@5: {final_top5:.2f}%")
#         metrics = {'top-1': final_top1, 'top-5': final_top5}
#
#         if args.save_logs:
#             for name, val in metrics.items():
#                 if tb_writer is not None:
#                     tb_writer.add_scalar(f"val/v_cls/{name}", val, epoch)
#
#             with open(os.path.join(args.output_dir, "results.jsonl"), "a+") as f:
#                 f.write(json.dumps(metrics))
#                 f.write("\n")
#
#         return metrics

def evaluate_v_cls(model, data, epoch, args, tb_writer=None):
    temp_val_v_cls_data = args.val_v_cls_data
    args.val_v_cls_data = list(data.keys())
    assert len(args.val_v_cls_data) == 1


    model.eval()
    dataloader = data[args.val_v_cls_data[0]]



    args.output_dir = os.path.join(args.log_base_path, f'video_cls/{args.val_v_cls_data[0].lower()}')
    os.makedirs(args.output_dir, exist_ok=True)
    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        if is_master(args):
            logging.info(f"Eval Epoch: {epoch}, accuracy of zero-shot classification under {args.val_v_cls_data[0].lower()} test videos")
        zero_shot_eval(model, dataloader, epoch, args)

    torch.distributed.barrier()

    if is_master(args):
        logging.info("Start merging results...")
        final_top1, final_top5 = merge(args.output_dir, args.world_size)
        logging.info(f"\t>>>  Acc@1: {final_top1:.2f}%, Acc@5: {final_top5:.2f}%")
        metrics = {'top-1': final_top1, 'top-5': final_top5}

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/v_cls/{args.val_v_cls_data[0].lower()}/{name}", val, epoch)

            with open(os.path.join(args.output_dir, "results.jsonl"), "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")

        args.val_v_cls_data = temp_val_v_cls_data
        return metrics

    args.val_v_cls_data = temp_val_v_cls_data
