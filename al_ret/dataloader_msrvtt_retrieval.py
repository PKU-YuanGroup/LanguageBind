from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os

import torchaudio
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random

from torchvision.io import read_video


class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            transform=77,
            max_words=30,
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.max_words = max_words
        self.tokenizer = tokenizer

        # self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.transform = transform
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}



    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            # words = self.tokenizer.tokenize(sentence)
            #
            # words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            # total_length_with_CLS = self.max_words - 1
            # if len(words) > total_length_with_CLS:
            #     words = words[:total_length_with_CLS]
            # words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            #
            # input_ids = self.tokenizer.convert_tokens_to_ids(words)
            # input_mask = [1] * len(input_ids)
            # segment_ids = [0] * len(input_ids)


            output = self.tokenizer(sentence)

            input_ids = output[0].squeeze()
            input_mask = output[1].squeeze()
            segment_ids = [0] * len(input_ids)


            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        # Pair x L x T x 3 x H x W
        audio = np.zeros((len(choice_video_ids), 3,
                          self.transform.num_mel_bins, self.transform.target_length), dtype=np.float)
        assert len(choice_video_ids) == 1
        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            # raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            # _, raw_audio_data, info = read_video(video_path, pts_unit='sec')
            # audio_data = self.transform((raw_audio_data, info['audio_fps']))

            audio_data = torchaudio.load(video_path.replace('mp4', 'wav'))
            audio_data = self.transform(audio_data)
            # audio[i] = audio_data
        return audio_data

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        audio_data = self._get_rawvideo(choice_video_ids)
        return audio_data, pairs_text, pairs_mask
