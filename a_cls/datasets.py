import os.path

import torch

from data.build_datasets import DataInfo
from data.process_audio import get_audio_transform, torchaudio_loader
from torchvision import datasets

# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import logging

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


class AudiosetDataset(Dataset):
    def __init__(self, args, transform, loader):
        self.audio_root = '/apdcephfs_cq3/share_1311970/downstream_datasets/Audio/audioset/eval_segments'
        dataset_json_file = '/apdcephfs_cq3/share_1311970/downstream_datasets/Audio/audioset/filter_eval.json'
        label_csv = '/apdcephfs_cq3/share_1311970/downstream_datasets/Audio/audioset/class_labels_indices.csv'
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)

        self.args = args
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0
        label_indices = torch.FloatTensor(label_indices)

        audio = self.loader(os.path.join(self.audio_root, datum['wav']))
        audio_data = self.transform(audio)
        return audio_data, label_indices

    def __len__(self):
        return len(self.data)



def is_valid_file(path):
    return True

def get_audio_dataset(args):
    data_path = args.audio_data_path
    transform = get_audio_transform(args)

    if args.val_a_cls_data.lower() == 'audioset':
        dataset = AudiosetDataset(args, transform=transform, loader=torchaudio_loader)
    else:
        dataset = datasets.ImageFolder(data_path, transform=transform, loader=torchaudio_loader, is_valid_file=is_valid_file)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    return DataInfo(dataloader=dataloader, sampler=None)
