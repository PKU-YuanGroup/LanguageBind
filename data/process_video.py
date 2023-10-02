
import io
import logging
import os

import cv2
import numpy as np
import torch
import decord
import torchvision.transforms
from PIL import Image
from decord import VideoReader, cpu

try:
    from petrel_client.client import Client
    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False


from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import sys
sys.path.append('../')
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from os.path import join as opj


def get_video_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    def _loader(video_path):
        if _client is not None and 's3:' in video_path:
            video_path = io.BytesIO(_client.get(video_path))

        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return vr

    return _loader


decord.bridge.set_bridge('torch')
# video_loader = get_video_loader()


def get_video_transform(args):
    if args.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(args.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    RandomCropVideo(size=224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif args.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                RandomCropVideo(size=224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif args.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                RandomCropVideo(size=224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif args.video_decode_backend == 'imgs':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                # Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                RandomCropVideo(size=224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv, imgs)')
    return transform

def load_and_transform_video(
    video_path,
    transform,
    video_decode_backend='opencv',
    clip_start_sec=0.0,
    clip_end_sec=None,
    num_frames=8,
):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'imgs':
        resize256_folder = video_path.replace('.mp4', '_resize256_folder')
        video_data = [ToTensor()(Image.open(opj(resize256_folder, f'{i}.jpg'))) for i in range(8)]
        video_data = torch.stack(video_data, dim=1)
        # print(video_data.shape, video_data.max(), video_data.min())
        video_outputs = transform(video_data)

    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv, imgs)')
    return {'pixel_values': video_outputs}

if __name__ == '__main__':
    load_and_transform_video(r"D:\ONE-PEACE-main\lb_test\zHSOYcZblvY.mp4")