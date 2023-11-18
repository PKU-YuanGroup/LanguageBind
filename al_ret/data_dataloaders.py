import argparse
import torch
from torch.utils.data import DataLoader

from data.build_datasets import get_data
from data.process_audio import get_audio_transform
from .dataloader_msrvtt_retrieval import MSRVTT_DataLoader

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        transform=get_audio_transform(args)
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"val":dataloader_msrvtt_test, "test":None}
