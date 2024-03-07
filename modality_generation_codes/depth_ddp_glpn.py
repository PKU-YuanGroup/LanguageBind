import sys

from PIL import Image
from torchvision import transforms
# from transformers import OFATokenizer, OFAModel
# from transformers.models.ofa.generate import sequence_generator  # from generate import sequence_generator
from transformers import DPTImageProcessor, DPTForDepthEstimation
import os.path
from argparse import ArgumentParser
from torch.utils import data
import json
import torch
import torch.distributed as dist
import os
import os.path as osp
from os.path import join as  opj
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

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

import decord
from decord import cpu
#glpn
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import io

import cv2
import numpy as np
from decord import VideoReader, cpu

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
        self.resolution = args.resolution # 对于动态大小视频无用
        self.loader = get_video_loader()

        if args.train_file.endswith('.csv'):
            self.train_file = pd.read_csv(args.train_file)
        elif args.train_file.endswith('.json'):
            # coco_vat_vat0_11_all_id_rootfolder_clsidx_spacy.json 
            # 格式：   id :  {   'idx_list' :  [0],  'root_folder'  :  'coco_vat_9' }

            if hasattr(args, 'part_nums') and args.part_nums >1:
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
                #=============================
                # obtain subset of self.id_list
                self.id_list = self.id_list[self.part_nums*(self.part_index-1):self.part_nums*self.part_index]


                print(f'Nums of train_file is {len(self.id_list)},part_index:{self.part_index}, first:{self.id_list[0]}')
                self.no_caption_id_list = []
                for idx,id in enumerate(self.id_list):
                    caption_json = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',self.train_file[id]['root_folder'],f'{id}_depth_f8glpn_folder')
                    mp4_path = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',self.train_file[id]['root_folder'],f'{id}.mp4')
                    if not os.path.exists(caption_json) and os.path.exists(mp4_path):
                        self.no_caption_id_list.append(mp4_path)
                    # else:
                    #     print(f'{caption_json} is exist!')
                    if idx%10000==0:
                        print(f'Time_cost:{time.time()-t1}s, idx:{idx}, caption_json:{caption_json}')
                try:
                    print(f'Nums of no_depth_folder_id_list is {len(self.no_caption_id_list)}, first:{self.no_caption_id_list[0]}')
                except:
                    print(f'Nums of no_depth_folder_id_list is {len(self.no_caption_id_list)}')
            t2 = time.time()
            print(f'Time cost:{t2-t1}s')
        # DPT
        # self.patch_resize_transform = DPTImageProcessor.from_pretrained("Intel/dpt-large", cache_dir= args.weights_folder)
        # glpn
        self.patch_resize_transform  = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu", cache_dir= args.weights_folder)
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

    def resize_frame(self, frame):
        height, width = frame.shape[:2]
        if height < width:
            new_height = 256
            new_width = int(width * (new_height / height))
        else:
            new_width = 256
            new_height = int(height * (new_width / width))
        # resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frame = cv2.resize(frame, (448, 796)) # 576*448   796,448
        return resized_frame, new_width,  new_height # frame的形状是和new_w, new_w不一样的！！


    def get_frames_from_video_decord(self, batchsize=1, video_path=None, caption_nums_per_video=8):
        # 加载视频
        video_path = video_path
        vr = self.loader(video_path)
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
            frame, new_width, new_height = self.resize_frame(frame)

            if i == 0:
                height, width = new_height, new_width

            # 转换为PIL Image并进行缩放
            # Image_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            Image_frame = Image.fromarray(frame)
            frame = self.patch_resize_transform(images=Image_frame, return_tensors="pt").pixel_values.unsqueeze(0)

            # 将numpy数组转换为tensor并存储在frames中
            frames.append(frame)

        # vr.close()
        frames = torch.cat(frames, 1).squeeze(0)
        return frames, height, width


    def get_frames_from_video_opencv(self, batchsize=1, video_path=None, caption_nums_per_video=8):
        # 加载视频
        video_path = video_path
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


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
            frame, new_width,  new_height = self.resize_frame(frame) # frame size是按照(w=448, h=576)resize的，但是new_h, new_w是按照原视频宽高比例缩放到短边为256，这样可以保持视频物体比例并且减少内存占用
            # print(f'{frame_width},{frame_height }===== {frame.shape} ')
            if i==0:
                # height, width, _ = frame.shape
                height, width =  new_height, new_width
                # print(f'{frame.shape}, {new_width},  {new_height}!!!')  #(576, 448, 3), 256,  455!!!
            if not ret:
                break
            # ret, frame = cap.read() # frame.shape (h,w,3)

            # frame = self.resize_frame(frame)
            
            # # print(f'{frame_width},{frame_height }===== {frame.shape} ')
            # if i==0:
            #     height, width, _ = frame.shape
            # if not ret:
            #     break

            # 转换为PIL Image并进行缩放
            Image_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # frame = self.patch_resize_transform(Image_frame)
            frame = self.patch_resize_transform(images=Image_frame, return_tensors="pt").pixel_values.unsqueeze(0)
            # print(frame.shape)

            # 将numpy数组转换为tensor并存储在frames中
            # frames[i] = frame
            frames.append(frame)

        # 打印输出frames的形状
        # print(frames.shape)
        cap.release()
        frames = torch.cat(frames, 1).squeeze(0)
        return frames, height, width


    

    def __getitem__(self, idx):

        try:
        
            # video_id = self.filter_train_file[idx]
            # video_path = opj(self.vat_root, video_id)
            video_path = self.no_caption_id_list[idx]
            video_id = video_path.split('/')[-1].split('.')[0]
            # 假如多个程序一起跑，其他已经生成了，就跳过
            caption_video_json =  video_path.replace('.mp4', '_depth_f8glpn_folder')
            if os.path.exists(caption_video_json):
                print('parallel task has process it :{}'.format(caption_video_json))
                # return '===========', None, torch.random(8,3,self.resolution, self.resolution)
                return self.skip_sample(idx)
            if not osp.exists(video_path):
                print('video {} is not exists and skip this idx! '.format(video_path))
                return self.skip_sample(idx)
            video_frames, height, width = self.get_frames_from_video_opencv( video_path = video_path, caption_nums_per_video = args.caption_nums_per_video)
            # video_frames, height, width = self.get_frames_from_video_decord( video_path = video_path, caption_nums_per_video = args.caption_nums_per_video)
            return video_id, video_path, video_frames, height, width
            
        except Exception as e:
            print('Read video error in {},{} and we have skip this !, this will not cause error!'.format(idx,e))
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


def depth_estimation(args):

    ########################################  model start #############################
    """https://huggingface.co/docs/transformers/main/en/model_doc/dpt"""
    
    weights_folder = args.weights_folder
    print(f'args.weights_folder is {args.weights_folder}')

    model_name = 'glpn'
    if model_name == 'glpn':
        # glpn
        feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu", cache_dir= weights_folder)
        model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu", cache_dir= weights_folder).cuda(args.local_rank)
    else:
        # DPT
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-large", cache_dir= weights_folder)
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large", cache_dir= weights_folder).cuda(args.local_rank)

    if args.rank==0:
        print('模型初始化完成')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                        output_device=args.local_rank)
    model.eval()
    if args.rank==0:
        print('DDP model')
    ########################################  model over #############################

    ######################################## dataset start #############################
    if args.rank == 0:
        print('dataset 初始化')

    train_dataset = my_dataset(args)
    if args.rank == 0:
        print('dataset_len: ',train_dataset.__len__())
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
    ######################################## depth_estimation start #############################   

    for index, (video_ids, video_paths,  videos_frames_, h_list, w_list) in enumerate(dataloader):
        
        bs, cap_nums, c, h, w = videos_frames_.shape
        videos_frames = videos_frames_.view(-1, c, h, w ).cuda(args.local_rank)
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                outputs = model(videos_frames)
                predicted_depth = outputs.predicted_depth # (bs*cap_nums, h, w)
            
            
            predicted_depth = predicted_depth.view(bs, cap_nums, h, w)
            # print(f'predicted_depth.shape:{predicted_depth.shape}')
            # interpolate to original size
            for bs_idx, sample in enumerate(predicted_depth):
                # import ipdb
                # ipdb.set_trace()
                pic_folder = video_paths[bs_idx].replace('.mp4','_depth_f8glpn_folder')
                os.makedirs(pic_folder, exist_ok=True)
                for frame_idx, frame in enumerate(predicted_depth[bs_idx]):
                    prediction = torch.nn.functional.interpolate(
                        frame.unsqueeze(0).unsqueeze(0), # torch.Size([1, 1, 384, 384])
                        size=(h_list[bs_idx],w_list[bs_idx]),
                        mode="bicubic",
                        align_corners=False,
                    )  # torch.Size([1, 1, h=480,  w=640])
                    # print('prediction.shape:{prediction.shape}')
                    # visualize the prediction
                    output = prediction.squeeze().cpu().numpy()
                    # formatted = (output * 255 / np.max(output)).astype("uint8")
                    # depth = Image.fromarray(formatted) # size (576, 1024)
                    # depth.save(f"{pic_folder}/{frame_idx}.png")
                    max_depth = 10
                    if np.any(output>10):
                        print(f"{pic_folder} > 10")
                    output_1k = np.clip(output, 0, max_depth)*1000
                    cv2.imwrite(f"{pic_folder}/{frame_idx}.png", output_1k.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(f'{pic_folder} is succeed!')
                # sys.exit(0)
            del videos_frames,  outputs, predicted_depth
        except Exception as e:
            print(f'Error:{e}!')
            del videos_frames
    ######################################## depth_estimation over  #############################  
def init_distributed_mode(args):

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)


    args.dist_backend = 'nccl'
    args.dist_url = 'env://'

    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=5400))
    torch.distributed.barrier()


import utils.misc as misc

def main(args):
    misc.init_distributed_mode(args)
    if args.rank == 0:
        print('进程组初始化完成')
        print("started")
        print("started caption_json count!")
        # glob1(json_path=args.exist_caption_id_list_json)  # 'coco_vat_exist_caption_id_list_03141026.json'
    ###########################################################3
    import time
    t1=time.time()
    depth_estimation(args)
    t2 = time.time()
    if args.rank == 0:
        print('Time : ',t2-t1,' s')
    dist.destroy_process_group()  # 销毁进程组
    
    
def test_dataset(args):

    train_dataset = my_dataset(args)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    from time import time
    for i, sample in enumerate(loader):
        video_ids, video_paths,  videos_frames, h, w = sample
        print(i, video_ids, video_paths, videos_frames.shape, h, w)
        
def glpn():

    from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
    import torch
    import numpy as np
    from PIL import Image
    import requests

    max_depth = 10

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open('hallo.png')
    weights_folder = './glpn'
    feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu", cache_dir= weights_folder)
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu", cache_dir= weights_folder)

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")
    # import ipdb
    # ipdb.set_trace()
    # print(inputs.shape)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    # formatted = (output * 255 / np.max(output)).astype("uint8")
    # depth = Image.fromarray(formatted)
    # print(f'min_output:{torch.min(output)}, max_output:{torch.max(output)}!!')
    output_1k = np.clip(output, 0, max_depth)*1000
    import ipdb
    ipdb.set_trace()
    cv2.imwrite('./saved_10000jpg.jpg', output_1k.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # cv2.imwrite('./image1.jpg', output_1k.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite('./saved_10000png.png', output_1k.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # cv2.imwrite('./image2.png', output_1k.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])


# glpn()
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
    parser.add_argument('--vat_root', type=str,default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=480)
    # parser.add_argument('--exist_caption_id_list_json', type=str, default=f'coco_vat_exist_caption_id_list_{month}{day}{hour}.json',help='')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')  # --dist_on_itp   ddp
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, help='url used to set up distributed training')
    parser.add_argument('--gpus', default=[0, 1, 2, 3], help='DP CUDA devices')
    parser.add_argument('--part_index', default=1, type=int, help='used to split train_file_id into different parts, and generate caption from part_index 1 to ....')
    parser.add_argument('--part_nums', default=1000, type=int, help='used to split train_file_id into different parts, and generate caption from part_index 1 to ....')
    parser.add_argument('--weights_folder', type=str,default='/apdcephfs_cq3/share_1311970/A_ofa/glpn')
    args = parser.parse_args()
    # test_dataset(args)
    # import ipdb
    # ipdb.set_trace()
    main(args)
    synchronize()
    # success_file=f"part_{args.part_index}_success"
    success_file=f"/apdcephfs_cq3/share_1311970/A_depth_glpn/part_{args.part_index}_success"
    os.system(f"touch {success_file}")

    """ 

    HF_DATASETS_OFFLINE=1 python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 29504  depth_ddp_glpn.py \
    --train_file  /apdcephfs_cq3/share_1311970/A_Youtube/coco_vat_vat0_11_all_id_rootfolder_clsidx_spacy.json   \
    --num_workers  1   --batch_size 2   \
    --part_index 92 \
    --part_nums  10000 \
    --weights_folder     /apdcephfs_cq3/share_1311970/A_ofa/glpn \
    --resolution 0 \
    --caption_nums_per_video 8
    
    """



