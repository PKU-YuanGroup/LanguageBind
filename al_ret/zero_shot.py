import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from .precision import get_autocast

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    # metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def _run_on_single_gpu(model, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    logit_scale = model.logit_scale.exp()
    for idx1, sequence_output in enumerate(batch_sequence_output_list):
        each_row = []
        for idx2, visual_output in enumerate(batch_visual_output_list):
            b1b2_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def run(model, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        sequence_output_list, visual_output_list = [], []
        for images, input_ids, attention_mask in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            images = images.unsqueeze(2)
            input_ids = input_ids.squeeze().to(args.device)
            attention_mask = attention_mask.squeeze().to(args.device)

            with autocast():
                # predict
                sequence_output = model.encode_text(input_ids, attention_mask)
                visual_output = model.encode_image(images)
            sequence_output_list.append(sequence_output)
            visual_output_list.append(visual_output)
    sim_matrix = _run_on_single_gpu(model, sequence_output_list, visual_output_list)
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    return sim_matrix


def zero_shot_eval(model, data, epoch, args):
    temp_val_al_ret_data = args.val_al_ret_data
    args.val_al_ret_data = list(data.keys())
    assert len(args.val_al_ret_data) == 1
    args.val_al_ret_data = args.val_al_ret_data[0]

    if args.val_al_ret_data not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info(f'Starting zero-shot {args.val_al_ret_data.upper()}.')

    results = {}
    if args.val_al_ret_data in data:
        logit_matrix = run(model, data[args.val_al_ret_data].dataloader, args)
        results = compute_metrics(logit_matrix)

    logging.info(f'Finished zero-shot {args.val_al_ret_data.upper()}.')

    args.val_al_ret_data = temp_val_al_ret_data
    return results
