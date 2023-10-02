import json

import torch
import numpy as np
import random
import os
import time
import argparse
import logging

from training.distributed import is_master
from .metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

# torch.distributed.init_process_group(backend="nccl")
from .util import parallel_apply


def _run_on_single_gpu(model,
                       # batch_list_t, batch_list_v,
                       batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1 in range(len(batch_sequence_output_list)):
        # input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2 in range(len(batch_visual_output_list)):
            # video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            # b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
            #                                                          loose_type=model.loose_type)
            # logging.info(f"{model.logit_scale.device}, {visual_output.device}, {sequence_output.device}")
            b1b2_logits = model.logit_scale * sequence_output @ visual_output.T
            # print(model.logit_scale.device, visual_output.device, sequence_output.device)
            # logging.info(f"{b1b2_logits.shape}, {b1b2_logits.device}")
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix



def evaluate_vl_ret(model, data, epoch, args, tb_writer=None):
    if is_master(args) and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # print(data)
        val_vl_ret_data = list(data.keys())
        # print(val_vl_ret_data)
        assert len(val_vl_ret_data) == 1
        val_vl_ret_data = val_vl_ret_data[0]
        test_dataloader = data[val_vl_ret_data]
        device = model.device
        n_gpu = torch.cuda.device_count()
        logging.info(f"\nEval Epoch: {epoch}, eval Video-Text Retrieval under {val_vl_ret_data.upper()} test data")
        if hasattr(model, 'module'):
            model = model.module.to(device)
        else:
            model = model.to(device)
        # #################################################################
        ## below variables are used to multi-sentences retrieval
        # multi_sentence_: important tag for eval
        # cut_off_points: used to tag the label when calculate the metric
        # sentence_num: used to cut the sentence representation
        # video_num: used to cut the video representation
        # #################################################################
        multi_sentence_ = False
        cut_off_points_, sentence_num_, video_num_ = [], -1, -1
        if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') and test_dataloader.dataset.multi_sentence_per_video:
            multi_sentence_ = True
            cut_off_points_ = test_dataloader.dataset.cut_off_points
            sentence_num_ = test_dataloader.dataset.sentence_num
            video_num_ = test_dataloader.dataset.video_num
            cut_off_points_ = [itm - 1 for itm in cut_off_points_]

        if multi_sentence_:
            logging.info("Eval under the multi-sentence per video clip setting.")
            logging.info("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

        model.eval()
        with torch.no_grad():
            # batch_list_t = []
            # batch_list_v = []
            batch_sequence_output_list, batch_visual_output_list = [], []
            total_video_num = 0

            # ----------------------------
            # 1. cache the features
            # ----------------------------
            for bid, batch in enumerate(test_dataloader):
                # batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, _, video, _ = batch
                # print(input_ids.shape, video.shape, video.dtype, end='-----')
                input_ids = input_ids.squeeze().to(device)
                attention_mask = attention_mask.squeeze().to(device)
                # video = video.squeeze().permute(0, 2, 1, 3, 4).float().to(device)
                video = video.float().to(device)



                # print(input_ids.shape, video.shape, video.dtype)
                # print(input_ids.shape, video.shape)
                if multi_sentence_:
                    # multi-sentences retrieval means: one clip has two or more descriptions.
                    b, *_t = video.shape
                    sequence_output = model.encode_text(input_ids, attention_mask)
                    # logging.info(f'multi: {sequence_output.shape}')
                    # sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                    batch_sequence_output_list.append(sequence_output)
                    # batch_list_t.append((input_mask, segment_ids,))

                    s_, e_ = total_video_num, total_video_num + b
                    filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                    if len(filter_inds) > 0:
                        # video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                        # print('before', video.shape)
                        video = video[filter_inds, ...]
                        # print('after', video.shape)
                        # visual_output = model.get_visual_output(video, video_mask)
                        visual_output = model.encode_image(video)
                        batch_visual_output_list.append(visual_output)
                        # batch_list_v.append((video_mask,))
                    total_video_num += b
                else:
                    sequence_output = model.encode_text(input_ids, attention_mask)
                    visual_output = model.encode_image(video)
                    # logging.info(f"{device}, {sequence_output.shape}, {visual_output.shape}")
                    # sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                    batch_sequence_output_list.append(sequence_output)
                    # batch_list_t.append((input_mask, segment_ids,))

                    batch_visual_output_list.append(visual_output)
                    # batch_list_v.append((video_mask,))

                print(f"Process {val_vl_ret_data.upper()}: {bid}/{len(test_dataloader)}\r", end='')

            # ----------------------------------
            # 2. calculate the similarity
            # ----------------------------------
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                device_ids = list(range(n_gpu))
                batch_t_output_splits = []
                batch_v_output_splits = []
                bacth_len = len(batch_sequence_output_list)
                # print(bacth_len)
                split_len = (bacth_len + n_gpu - 1) // n_gpu
                for dev_id in device_ids:
                    s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                    if dev_id == 0:

                        batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                        batch_v_output_splits.append(batch_visual_output_list)
                        # print(len(batch_sequence_output_list[s_:e_]), len(batch_visual_output_list))
                    else:
                        devc = torch.device('cuda:{}'.format(str(dev_id)))

                        devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                        batch_t_output_splits.append(devc_batch_list)
                        devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                        batch_v_output_splits.append(devc_batch_list)
                        # print(len(devc_batch_list), len(devc_batch_list))
                parameters_tuple_list = [(
                                          batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
                parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
                sim_matrix = []
                for idx in range(len(parallel_outputs)):
                    sim_matrix += parallel_outputs[idx]
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            else:
                sim_matrix = _run_on_single_gpu(model,
                                                # batch_list_t, batch_list_v,
                                                batch_sequence_output_list, batch_visual_output_list)
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        #####################################################################
        if multi_sentence_:
            logging.info(f"{val_vl_ret_data.upper()} before reshape, sim matrix size: {sim_matrix.shape}")
            cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
            max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
            sim_matrix_new = []
            for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                      np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
            sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
            logging.info(f"{val_vl_ret_data.upper()} after reshape, sim matrix size: {sim_matrix.shape}")

            tv_metrics = tensor_text_to_video_metrics(sim_matrix)
            vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
        else:
            logging.info(f"{val_vl_ret_data.upper()} sim matrix size: {sim_matrix.shape[0]}, {sim_matrix.shape[1]}")
            tv_metrics = compute_metrics(sim_matrix)
            vt_metrics = compute_metrics(sim_matrix.T)
            logging.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

        logging.info(f"{val_vl_ret_data.upper()} Text-to-Video:")
        logging.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        logging.info(f"{val_vl_ret_data.upper()} Video-to-Text:")
        logging.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))


        if args.save_logs:
            for name, val in tv_metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/vl_ret/{val_vl_ret_data}/t2v/{name}", val, epoch)
            for name, val in vt_metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/vl_ret/{val_vl_ret_data}/v2t/{name}", val, epoch)

            args.vl_ret_output_dir = os.path.join(args.log_base_path, f'vl_ret/{val_vl_ret_data}')
            os.makedirs(args.vl_ret_output_dir, exist_ok=True)
            with open(os.path.join(args.vl_ret_output_dir, "results.jsonl"), "a+") as f:
                f.write(json.dumps({'t2v': tv_metrics}))
                f.write("\n")
                f.write(json.dumps({'v2t': vt_metrics}))
                f.write("\n")

        # R1 = tv_metrics['R1']
        # return R1

    # torch.distributed.barrier()