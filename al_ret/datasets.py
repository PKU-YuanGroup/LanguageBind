import logging
import os.path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data.build_datasets import DataInfo
from open_clip import get_input_dtype, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from data.process_audio import get_audio_transform, torchaudio_loader

class Audiocaps_dataset(Dataset):
    def __init__(self, data_path, transform, loader, tokenizer):
        super(Audiocaps_dataset, self).__init__()
        self.audio_root = data_path
        raw_meta = pd.read_csv(f'{self.audio_root}/audiocaps_test.tsv', delimiter='\t').values
        audio_ids = list(set(raw_meta[:, 1].tolist()))
        captions = {}
        for i in raw_meta:
            if captions.get(i[1], None) is None:
                captions[i[1]] = [i[2]]
            else:
                captions[i[1]] = captions[i[1]] + [i[2]]
        # captions = {i[:1][0]: i[1:].tolist() for i in raw_meta}


        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for audio_id in audio_ids:
            assert audio_id in captions
            for cap in captions[audio_id]:
                cap_txt = cap
                self.sentences_dict[len(self.sentences_dict)] = (audio_id[10:], cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        self.multi_sentence_per_audio = True  # !!! important tag for eval
        if self.multi_sentence_per_audio:
            # if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.audio_num = len(audio_ids)
            assert len(self.cut_off_points) == self.audio_num
            print("Sentence number: {}".format(self.sentence_num))
            print("Video number: {}".format(self.audio_num))

        self.sample_len = len(self.sentences_dict)

        self.transform = transform
        self.torchaudio_loader = loader
        self.tokenizer = tokenizer

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        audiocap_id, caption = self.sentences_dict[idx]

        audio_path = os.path.join(self.audio_root, audiocap_id)
        audio = self.torchaudio_loader(audio_path)
        audio_data = self.transform(audio)

        input_ids, attention_mask = self.tokenizer(caption)
        return audio_data, input_ids.squeeze(), attention_mask.squeeze()


class Clotho_dataset(Dataset):
    def __init__(self, data_path, transform, loader, tokenizer):
        super(Clotho_dataset, self).__init__()
        self.audio_root = data_path
        raw_meta = pd.read_csv(f'{self.audio_root}/CLOTHO_retrieval_dataset/clotho_captions_evaluation.csv').values
        audio_ids = raw_meta[:, 0].tolist()
        captions = {i[:1][0]: i[1:].tolist() for i in raw_meta}
        # self.meta = pd.DataFrame(np.vstack([np.vstack([raw_meta[:, 0], raw_meta[:, i]]).T for i in range(1, 6)]),
        #                          columns=['uniq_id', 'text'])

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for audio_id in audio_ids:
            assert audio_id in captions
            for cap in captions[audio_id]:
                cap_txt = cap
                self.sentences_dict[len(self.sentences_dict)] = (audio_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        self.multi_sentence_per_audio = True    # !!! important tag for eval
        if self.multi_sentence_per_audio:
            # if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.audio_num = len(audio_ids)
            assert len(self.cut_off_points) == self.audio_num
            print("Sentence number: {}".format(self.sentence_num))
            print("Video number: {}".format(self.audio_num))

        self.sample_len = len(self.sentences_dict)

        self.transform = transform
        self.torchaudio_loader = loader
        self.tokenizer = tokenizer

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        audiocap_id, caption = self.sentences_dict[idx]
        # audiocap_id = self.meta['uniq_id'][idx]
        audio_path = os.path.join(self.audio_root, f'evaluation/{audiocap_id}')
        audio = self.torchaudio_loader(audio_path)
        audio_data = self.transform(audio)

        # caption = self.meta['text'][idx]
        input_ids, attention_mask = self.tokenizer(caption)
        return audio_data, input_ids.squeeze(), attention_mask.squeeze()

def get_audio_dataset(args):
    data_path = args.audio_data_path
    transform = get_audio_transform(args)
    tokenizer = get_tokenizer(HF_HUB_PREFIX+args.model, cache_dir=args.cache_dir)

    if args.val_al_ret_data.lower() == 'audiocaps':
        dataset = Audiocaps_dataset(data_path, transform=transform, loader=torchaudio_loader, tokenizer=tokenizer)
    elif args.val_al_ret_data.lower() == 'clotho':
        dataset = Clotho_dataset(data_path, transform=transform, loader=torchaudio_loader, tokenizer=tokenizer)
    else:
        raise ValueError(f'unsupport dataset {args.val_al_ret_data}')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
    )

    return dataloader
