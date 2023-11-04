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

def make_midname_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['display_name']
            line_count += 1
    return index_lookup

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)

    def __getitem__(self, index):
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        try:
            fbank, mix_lambda = self._wav2fbank(datum['wav'])
        except Exception as e:
            logging.warning(f"Error at {datum['wav']} with \"{e}\"")
            return self.__getitem__(random.randint(0, self.__len__()-1))
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)


        return fbank, label_indices

    def __len__(self):
        return len(self.data)