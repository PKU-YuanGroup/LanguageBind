import os
import time
from dataclasses import dataclass
from multiprocessing import Value

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.base_datasets import VAT_dataset
from data.new_loadvat import get_wds_dataset
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def get_VAT_dataset(args):
    dataset = VAT_dataset(args)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # prefetch_factor=2,
        # persistent_workers=True,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_data(args, epoch=0):
    data = {}

    if args.do_train:
        if args.train_data.endswith(".json"):
            data[f"{args.clip_type}_pt"] = get_VAT_dataset(args)
        elif args.train_data.endswith(".tar"):
            data[f"{args.clip_type}_pt"] = get_wds_dataset(args, is_train=True, epoch=epoch)
        else:
            raise NameError

    if args.do_eval:
        temp_batch_size = args.batch_size
        args.batch_size = 8 if args.val_vl_ret_data else 16
        data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/VideoTextRetrieval/vtRetdata"
        if args.val_vl_ret_data:
            data["vl_ret"] = []
            for val_vl_ret_data in args.val_vl_ret_data:
                if val_vl_ret_data == "msrvtt":
                    args.train_csv = os.path.join(f'{data_root}/MSRVTT/MSRVTT_train.9k.csv')
                    args.val_csv = os.path.join(f'{data_root}/MSRVTT/MSRVTT_JSFUSION_test.csv')
                    args.data_path = os.path.join(f'{data_root}/MSRVTT/MSRVTT_data.json')
                    args.features_path = os.path.join(f'{data_root}/MSRVTT/MSRVTT_Videos')
                elif val_vl_ret_data == "msvd":
                    args.data_path = os.path.join(f'{data_root}/MSVD')
                    args.features_path = os.path.join(f'{data_root}/MSVD/MSVD_Videos')
                elif val_vl_ret_data == "activity":
                    args.data_path = os.path.join(f'{data_root}/ActivityNet')
                    args.features_path = os.path.join(f'{data_root}/ActivityNet/Videos/Activity_Videos')
                elif val_vl_ret_data == "didemo":
                    args.data_path = os.path.join(f'{data_root}/Didemo')
                    args.features_path = os.path.join(f'{data_root}/Didemo/videos')
                else:
                    raise NameError

                args.batch_size_val = args.batch_size if args.batch_size_val == 0 else args.batch_size_val
                args.max_frames = args.num_frames
                args.num_thread_reader = args.workers
                args.slice_framepos = 2   # "0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly."

                from vl_ret.data_dataloaders import DATALOADER_DICT

                tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
                test_dataloader, test_length = None, 0
                if DATALOADER_DICT[val_vl_ret_data]["test"] is not None:
                    test_dataloader, test_length = DATALOADER_DICT[val_vl_ret_data]["test"](args, tokenizer)

                if DATALOADER_DICT[val_vl_ret_data]["val"] is not None:
                    val_dataloader, val_length = DATALOADER_DICT[val_vl_ret_data]["val"](args, tokenizer, subset="val")
                else:
                    val_dataloader, val_length = test_dataloader, test_length
                ## report validation results if the ["test"] is None
                if test_dataloader is None:
                    test_dataloader, test_length = val_dataloader, val_length

                data["vl_ret"].append({val_vl_ret_data: test_dataloader})

        if args.val_v_cls_data:
            from v_cls import get_video_cls_dataloader
            args.data_set = args.val_v_cls_data
            args.num_workers = args.workers
            args.num_sample = 1  # no repeat
            data["v_cls"] = get_video_cls_dataloader(args)


        if args.val_a_cls_data:
            data["a_cls"] = []
            data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/Audio"
            temp_val_a_cls_data = args.val_a_cls_data
            for val_a_cls_data in temp_val_a_cls_data:
                from a_cls.datasets import get_audio_dataset
                args.val_a_cls_data = val_a_cls_data
                args.audio_data_path = os.path.join(data_root, f'{val_a_cls_data.lower()}/test')
                data['a_cls'].append({val_a_cls_data: get_audio_dataset(args)})
            args.val_a_cls_data = temp_val_a_cls_data

        if args.imagenet_val is not None:
            from i_cls.datasets import get_imagenet
            data['i_cls'] = {}
            data['i_cls']["imagenet-val"] = get_imagenet(args, "val")
        if args.imagenet_v2 is not None:
            from i_cls.datasets import get_imagenet
            if data.get('i_cls', None) is None:
                data['i_cls'] = {}
            data['i_cls']["imagenet-v2"] = get_imagenet(args, "v2")

        if args.val_d_cls_data:
            data["d_cls"] = []
            data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/Depth"
            temp_val_d_cls_data = args.val_d_cls_data
            for val_d_cls_data in temp_val_d_cls_data:
                from d_cls.datasets import get_depth_dataset
                args.val_d_cls_data = val_d_cls_data
                args.depth_data_path = os.path.join(data_root, f'{val_d_cls_data.lower()}/data/val')
                data['d_cls'].append({val_d_cls_data: get_depth_dataset(args)})
            args.val_d_cls_data = temp_val_d_cls_data


        if args.val_t_cls_data:
            data["t_cls"] = []
            data_root = "/apdcephfs_cq3/share_1311970/downstream_datasets/Thermal"
            temp_val_t_cls_data = args.val_t_cls_data
            for val_t_cls_data in temp_val_t_cls_data:
                from t_cls.datasets import get_thermal_dataset
                args.val_t_cls_data = val_t_cls_data
                args.thermal_data_path = os.path.join(data_root, f'{val_t_cls_data.lower()}/val')
                data['t_cls'].append({val_t_cls_data: get_thermal_dataset(args)})
            args.val_t_cls_data = temp_val_t_cls_data

        args.batch_size = temp_batch_size

    return data



