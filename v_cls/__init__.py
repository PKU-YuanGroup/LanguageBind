import os

import torch
from functools import partial
from .build import build_dataset, build_pretraining_dataset
from torch.utils.data._utils.collate import default_collate

__all__ = ['build_dataset', 'build_pretraining_dataset']


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data

def get_video_cls_dataloader(args):
    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    # if args.disable_eval_during_finetuning:
    #     dataset_val = None
    # else:
    dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

    num_tasks = args.world_size
    global_rank = args.rank
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    # print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        # batch_size=16,                  ######################################
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=True)

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            # batch_size=16,                 ####################################
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True)
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            # batch_size=16,                 #####################################
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True)
    else:
        data_loader_test = None

    # return data_loader_train, data_loader_val, data_loader_test
    return data_loader_test