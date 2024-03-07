
from __future__ import print_function
from PIL import Image
from torchvision import transforms
# from transformers import OFATokenizer, OFAModel
# from transformers.models.ofa.generate import sequence_generator  # from generate import sequence_generator
import os.path
from argparse import ArgumentParser
from torch.utils import data
import json
import torch
import torch.distributed as dist
import os
import os.path as osp
from os.path import join as opj
import pandas as pd
from random import randint
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import decord
import glob
import subprocess
import time
import numpy as np

from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from trainer import MUNIT_Trainer, UNIT_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
import io

import cv2
import numpy as np
from decord import VideoReader, cpu

import decord
from decord import cpu
import torch
import numpy as np
from PIL import Image
import requests
try:
    from petrel_client.client import Client
    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False


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


class my_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.shuffle = True
        self.resolution = args.resolution  # 对于动态大小视频无用
        self.video_loader = get_video_loader()
        if args.train_file.endswith('.csv'):
            self.train_file = pd.read_csv(args.train_file)
        elif args.train_file.endswith('.json'):
            # coco_vat_vat0_11_all_id_rootfolder_clsidx_spacy.json
            # 格式：   id :  {   'idx_list' :  [0],  'root_folder'  :  'coco_vat_9' }

            if hasattr(args, 'part_nums') and args.part_nums > 1:
                self.part_nums = args.part_nums
            else:
                self.part_nums = 100000
            self.part_index = args.part_index
            t1 = time.time()
            with open(args.train_file, 'r', encoding='utf-8') as f:
                self.train_file = json.load(f)
                if type(self.train_file) is str:
                    self.train_file = json.loads(self.train_file)

                self.id_list = list(self.train_file.keys())
                # =============================
                # obtain subset of self.id_list
                self.id_list = self.id_list[self.part_nums * (self.part_index - 1):self.part_nums * self.part_index]

                print(
                    f'Nums of train_file is {len(self.id_list)},part_index:{self.part_index}, first:{self.id_list[0]}')
                self.no_caption_id_list = []
                for idx, id in enumerate(self.id_list):
                    caption_json = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',
                                            self.train_file[id]['root_folder'], f'{id}_thermal_folder')
                    mp4_path = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube', self.train_file[id]['root_folder'],
                                        f'{id}.mp4')
                    if not os.path.exists(caption_json) and os.path.exists(mp4_path):
                        self.no_caption_id_list.append(mp4_path)
                    # else:
                    #     print(f'{caption_json} is exist!')
                    if idx % 10000 == 0:
                        print(f'Time_cost:{time.time() - t1}s, idx:{idx}, caption_json:{caption_json}')
                try:
                    print(f'Nums of no_thermal_folder_id_list is {len(self.no_caption_id_list)}, first:{self.no_caption_id_list[0]}')
                except:
                    print(f'Nums of no_thermal_folder_id_list is {len(self.no_caption_id_list)}')
                    
            t2 = time.time()
            print(f'Time cost:{t2 - t1}s')
        # DPT
        # self.patch_resize_transform = DPTImageProcessor.from_pretrained("Intel/dpt-large", cache_dir= args.weights_folder)
        # glpn
        self.patch_resize_transform = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5),
                                                                               (0.5, 0.5, 0.5)),
                                                          transforms.Resize((400, 640))
                                                          ])
        print('Dataset nums is {}'.format(self.__len__()))
        time.sleep(10)

    def __len__(self):
        return len(self.no_caption_id_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, idx):
        if idx >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(idx + 1)

    def skip_sample(self, idx):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(idx=idx)

    def get_frames_from_video_opencv(self, batchsize=1, video_path=None, caption_nums_per_video=8):
        # 加载视频
        video_path = video_path
        cap = cv2.VideoCapture(video_path)

        # 确定要提取的帧数
        num_frames = caption_nums_per_video
        # 计算每隔多少帧提取一次
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames // num_frames

        # 用于存储提取的图像的tensor
        # frames = torch.empty(num_frames, 3, frame_height, frame_width)
        frames = []
        height = width = 0
        # frame_idx_list=[]  把视频的帧序号存储下来
        # 直接读取指定帧
        for i in range(num_frames):
            # 计算要提取的帧的索引
            idx = i * step
            # frame_idx_list.append(idx)
            # 设置当前帧为所需的帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            # 读取该帧
            ret, frame = cap.read()
            if i == 0:
                height, width = frame.shape[:2]
                if height < width:
                    new_height = 256
                    new_width = int(width * (new_height / height))
                else:
                    new_width = 256
                    new_height = int(height * (new_width / width))
            if not ret:
                break
            # 转换为PIL Image并进行缩放
            Image_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = self.patch_resize_transform(Image_frame).unsqueeze(0).unsqueeze(0)
            # 将numpy数组转换为tensor并存储在frames中
            # frames[i] = frame
            # 将numpy数组转换为tensor并存储在frames中
            frames.append(frame)

        # vr.close()
        frames = torch.cat(frames, 1).squeeze(0)
        return frames, new_height, new_width


    def get_frames_from_video_decord(self, batchsize=1, video_path=None, caption_nums_per_video=8):
        # 加载视频
        video_path = video_path
        vr = self.video_loader(video_path)
        frame_width, frame_height = vr[0].shape[1], vr[0].shape[0]

        # 确定要提取的帧数
        num_frames = caption_nums_per_video
        # 计算每隔多少帧提取一次
        total_frames = len(vr)
        step = total_frames // num_frames

        # 用于存储提取的图像的tensor
        frames = []
        height = width = 0

        # 直接读取指定帧
        for i in range(num_frames):
            # 计算要提取的帧的索引
            idx = i * step
            # 读取该帧
            frame = vr[idx].asnumpy()
            if i == 0:
                height, width = frame.shape[:2]
                if height < width:
                    new_height = 256
                    new_width = int(width * (new_height / height))
                else:
                    new_width = 256
                    new_height = int(height * (new_width / width))

            # 转换为PIL Image并进行缩放
            Image_frame = Image.fromarray(frame)
            frame = self.patch_resize_transform(Image_frame).unsqueeze(0).unsqueeze(0)

            # 将numpy数组转换为tensor并存储在frames中
            frames.append(frame)

        # vr.close()
        frames = torch.cat(frames, 1).squeeze(0)
        return frames, new_height, new_width

    def __getitem__(self, idx):

        try:

            # video_id = self.filter_train_file[idx]
            # video_path = opj(self.vat_root, video_id)
            video_path = self.no_caption_id_list[idx]
            video_id = video_path.split('/')[-1].split('.')[0]
            # 假如多个程序一起跑，其他已经生成了，就跳过
            caption_video_json = video_path.replace('.mp4', '_thermal_folder')
            if os.path.exists(caption_video_json):
                print('parallel task has process it :{}'.format(caption_video_json))
                # return '===========', None, torch.random(8,3,self.resolution, self.resolution)
                return self.skip_sample(idx)
            if not osp.exists(video_path):
                print('video {} is not exists and skip this idx! '.format(video_path))
                return self.skip_sample(idx)
            # video_frames, height, width = self.get_frames_from_video_opencv( video_path = video_path, caption_nums_per_video = args.caption_nums_per_video)
            video_frames, height, width = self.get_frames_from_video_opencv(video_path=video_path,
                                                                            caption_nums_per_video=args.caption_nums_per_video)
            return video_id, video_path, video_frames, height, width

        except Exception as e:
            print('Read video error in {},{} and we have skip this !, this will not cause error!'.format(idx, e))
            return self.skip_sample(idx)


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()



def thermal_estimation(args):
    ########################################  model start #############################
    """https://huggingface.co/docs/transformers/main/en/model_doc/dpt"""


    config = 'configs/tir2rgb_folder.yaml'
    a2b = 0
    checkpoint = './translation_weights.pt'
    output_path = '.'
    num_style = 1
    config = get_config(config)
    config['vgg_model_path'] = output_path
    style_dim = config['gen']['style_dim']

    trainer = MUNIT_Trainer(config)

    state_dict = torch.load(checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

    trainer.cuda(args.local_rank)
    trainer.train()

    if args.rank == 0:
        print('模型初始化完成')
    trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    encode = trainer.module.gen_a.encode if a2b else trainer.module.gen_b.encode # encode function
    decode = trainer.module.gen_b.decode if a2b else trainer.module.gen_a.decode # decode function



    if args.rank == 0:
        print('DDP model')
    ########################################  model over #############################

    ######################################## dataset start #############################
    if args.rank == 0:
        print('dataset 初始化')

    train_dataset = my_dataset(args)
    if args.rank == 0:
        print('dataset_len: ', train_dataset.__len__())
        print('loading dataset is complete!')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank
    )
    if args.rank == 0:
        print('正在同步')
    synchronize()
    if args.rank == 0:
        print('dataloader 初始化')
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             sampler=train_sampler,
                                             drop_last=True
                                             )
    ######################################## dataset over  #############################
    ######################################## thermal_estimation start #############################

    for index, (video_ids, video_paths, videos_frames_, h_list, w_list) in enumerate(dataloader):
        # print(videos_frames_.shape)
        bs, cap_nums, c, h, w = videos_frames_.shape
        videos_frames = videos_frames_.reshape(-1, c, h, w)
        images = Variable(videos_frames.cuda(args.local_rank), volatile=True)
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                predicted_thermal = []
                for i in range(bs*cap_nums):
                    # print('images[i]', images[i].unsqueeze(0).shape)
                    content, _ = encode(images[i].unsqueeze(0))
                    # print('content', content.shape)
                    # style = style_fixed if opts.synchronized else Variable(torch.randn(num_style, style_dim, 1, 1).cuda(), volatile=False)
                    style = Variable(torch.randn(num_style, style_dim, 1, 1).cuda(args.local_rank), volatile=False)
                    s = style[0].unsqueeze(0)
                    # print('s', s.shape)
                    outputs = decode(content, s)
                    outputs = (outputs + 1) / 2.
                    # print('outputs', outputs.shape)


                    predicted_thermal.append(outputs)
            predicted_thermal = torch.cat(predicted_thermal, dim=0) # (bs*cap_nums, 3, h, w)
            predicted_thermal = predicted_thermal.view(bs, cap_nums, c, h, w)
            # print(f'predicted_thermal.shape:{predicted_thermal.shape}')
            # interpolate to original size
            for bs_idx, sample in enumerate(predicted_thermal):
                # import ipdb
                # ipdb.set_trace()
                pic_folder = video_paths[bs_idx].replace('.mp4', '_thermal_folder')
                os.makedirs(pic_folder, exist_ok=True)
                for frame_idx, frame in enumerate(predicted_thermal[bs_idx]):
                    # print(frame.shape, h_list, w_list)
                    prediction = torch.nn.functional.interpolate(
                        frame.unsqueeze(0),  # torch.Size([1, 3, 400, 640])
                        size=(h_list[bs_idx], w_list[bs_idx]),
                        mode="bicubic",
                        align_corners=False,
                    )  # torch.Size([1, 1, h=480,  w=640])
                    # print('prediction.shape:{prediction.shape}')

                    vutils.save_image(prediction.data, f"{pic_folder}/{frame_idx}.jpg", padding=0, normalize=True)


                print(f'{pic_folder} is succeed!')
                # sys.exit(0)
            del videos_frames, outputs, predicted_thermal
        except Exception as e:
            print(f'Error:{e}!')
            del videos_frames
    ######################################## thermal_estimation over  #############################


def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    args.dist_backend = 'nccl'
    args.dist_url = 'env://'

    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                         timeout=datetime.timedelta(seconds=5400))
    torch.distributed.barrier()


import misc


def main(args):
    misc.init_distributed_mode(args)
    if args.rank == 0:
        print('进程组初始化完成')
        print("started")
        print("started caption_json count!")
        # glob1(json_path=args.exist_caption_id_list_json)  # 'coco_vat_exist_caption_id_list_03141026.json'
    ###########################################################3
    import time
    t1 = time.time()
    thermal_estimation(args)
    t2 = time.time()
    if args.rank == 0:
        print('Time : ', t2 - t1, ' s')
    dist.destroy_process_group()  # 销毁进程组


def test_dataset(args):
    train_dataset = my_dataset(args)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    from time import time
    for i, sample in enumerate(loader):
        video_ids, video_paths, videos_frames, h, w = sample
        print(i, video_ids, video_paths, videos_frames.shape, h, w)


if __name__ == "__main__":
    import time
    # time.sleep(10000)
    import datetime

    # 获取当前时间
    now = datetime.datetime.now()
    # 获取当前月份
    month = now.month
    # 获取当前日期
    day = now.day
    # 获取当前小时
    hour = now.hour

    parser = ArgumentParser()
    parser.add_argument('--caption_nums_per_video', type=int, default=8, help='process rank')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--vat_root', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=480)
    # parser.add_argument('--exist_caption_id_list_json', type=str, default=f'coco_vat_exist_caption_id_list_{month}{day}{hour}.json',help='')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')  # --dist_on_itp   ddp
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, help='url used to set up distributed training')
    parser.add_argument('--gpus', default=[0, 1, 2, 3], help='DP CUDA devices')
    parser.add_argument('--part_index', default=1, type=int,
                        help='used to split train_file_id into different parts, and generate caption from part_index 1 to ....')
    parser.add_argument('--part_nums', default=1000, type=int,
                        help='used to split train_file_id into different parts, and generate caption from part_index 1 to ....')
    parser.add_argument('--weights_folder', type=str, default='/apdcephfs_cq3/share_1311970/A_thermal/')
    args = parser.parse_args()
    # test_dataset(args)
    # import ipdb
    # ipdb.set_trace()
    main(args)
    synchronize()
    # success_file=f"part_{args.part_index}_success"
    success_file = f"/apdcephfs_cq3/share_1311970/A_thermal/part_{args.part_index}_success"
    os.system(f"touch {success_file}")

    """ 
    cd /apdcephfs_cq3/share_1311970/A_thermal/sRGB-TIR
    source /apdcephfs_cq3/share_1311970/lb/miniconda3/etc/profile.d/conda.sh
    conda activate /apdcephfs_cq3/share_1311970/lb/miniconda3/envs/pytorch1.12.1
    export CUDA_VISIBLE_DEVICES=1
    HF_DATASETS_OFFLINE=1 python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 29504  thermal_ddp.py \
    --train_file  /apdcephfs_cq3/share_1311970/A_Youtube/coco_vat_vat0_11_all_id_rootfolder_clsidx_spacy.json   \
    --num_workers  8   --batch_size 1   \
    --part_index 91 \
    --part_nums  10000 \
    --resolution 0 \
    --caption_nums_per_video 8

    """
