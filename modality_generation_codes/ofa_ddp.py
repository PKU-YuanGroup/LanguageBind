
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator  # from generate import sequence_generator
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





class my_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.shuffle = True
        self.resolution = args.resolution

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
                # obtain subset of self.id_list, so that deduplication time is less than 30min 
                self.id_list = self.id_list[self.part_nums*(self.part_index-1):self.part_nums*self.part_index]


                print(f'Nums of train_file is {len(self.id_list)},part_index:{self.part_index}, first:{self.id_list[0]}')
                self.no_caption_id_list = []
                self.exist_caption_path_list = {}
                for idx, id in enumerate(self.id_list):
                    caption_json = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',self.train_file[id]['root_folder'],f'{id}_caption.json')
                    mp4_path = osp.join('/apdcephfs_cq3/share_1311970/A_Youtube',self.train_file[id]['root_folder'],f'{id}.mp4')
                    "existcap的数目包括video不存在的,所以有点虚大"
                    if not os.path.exists(caption_json) and os.path.exists(mp4_path):
                        self.no_caption_id_list.append(mp4_path)
                    else:
                        self.exist_caption_path_list[caption_json]=True

                    if idx%10000==0:
                        print(f'Time_cost:{time.time()-t1}s, idx:{idx}, caption_json:{caption_json}')
                print(f'Nums of no_caption_id_list is {len(self.no_caption_id_list)}, first:{self.no_caption_id_list[0]}')
                print(f'Nums of exist_caption_path_list is {len(self.exist_caption_path_list)}')
                if args.rank==0:
                    success_file=f"part_{args.part_index}_success_nocap_{len(self.no_caption_id_list)}_existcap{len(self.exist_caption_path_list)}"
                    os.system(f"touch {success_file}")
            t2 = time.time()
            print(f'Time cost:{t2-t1}s')
        # print('======',self.exist_file_list,'====',self.no_caption_id_list)

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.patch_resize_transform = transforms.Compose([
                        lambda image: image.convert("RGB"),
                        transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean=mean, std=std)
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
        frames = torch.empty(num_frames, 3, self.resolution, self.resolution)

        # 直接读取指定帧
        for i in range(num_frames):
            # 计算要提取的帧的索引
            idx = i * step
            # 设置当前帧为所需的帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            # 读取该帧
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为PIL Image并进行缩放
            Image_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = self.patch_resize_transform(Image_frame)

            # 将numpy数组转换为tensor并存储在frames中
            frames[i] = frame

        # 打印输出frames的形状
        # print(frames.shape)
        cap.release()
        return frames


    def get_frames_from_video(self, batchsize=1, video_path=None, caption_nums_per_video = 8, ):

        # 加载视频
        video_path = video_path
        vr = decord.VideoReader(video_path)
        # 确定要提取的帧数
        num_frames = caption_nums_per_video
        # 计算每隔多少帧提取一次
        step = len(vr) // num_frames
        # 用于存储提取的图像的tensor
        frames = torch.empty(num_frames, 3, self.resolution, self.resolution)

        # 从视频中提取图像
        for i in range(num_frames):
            # 计算要提取的帧的索引
            idx = i * step
            # 从视频中读取帧
            decord_frame = vr[idx].asnumpy()
            Image_frame = Image.fromarray(decord_frame)
            frame = self.patch_resize_transform(Image_frame)#.unsqueeze(0)
            
            # 将numpy数组转换为tensor并存储在frames中
            frames[i] = frame

        # 打印输出frames的形状
        # print(frames.shape)
        vr.close()
        return frames


    def __getitem__(self, idx):

        try:
        
            # video_id = self.filter_train_file[idx]
            # video_path = opj(self.vat_root, video_id)
            video_path = self.no_caption_id_list[idx]
            video_id = video_path.split('/')[-1].split('.')[0]
            # 假如多个程序一起跑，其他已经生成了，就跳过
            caption_video_json =  video_path.replace('.mp4', '_caption.json')
            if  caption_video_json in self.exist_caption_path_list:
                print('parallel task has process it :{}, this is duplication!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(caption_video_json))
                # return '===========', None, torch.random(8,3,self.resolution, self.resolution)
                # return self.skip_sample(idx)
            # if os.path.exists(caption_video_json):
            #     print('parallel task has process it :{}'.format(caption_video_json))
            #     # return '===========', None, torch.random(8,3,self.resolution, self.resolution)
            #     return self.skip_sample(idx)
            if not osp.exists(video_path):
                print('video {} is not exists and skip this idx! '.format(video_path))
                return self.skip_sample(idx)
            video_frames = self.get_frames_from_video_opencv( video_path = video_path, caption_nums_per_video = args.caption_nums_per_video)
            
            return video_id, video_path, video_frames
            
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


def ids_captions_save(args, video_ids, video_paths, caption_list):
    for i, video_path in enumerate(video_paths):
        caption_video_json =  video_path.replace('.mp4', '_caption.json')
        
        video_16captions = caption_list[i * args.caption_nums_per_video : (i+1) * args.caption_nums_per_video]
        video_caption_dict = { video_ids[i] : video_16captions }
        
        if osp.exists(caption_video_json):
            print('{} is exist, please check your train file'.format(caption_video_json))
            continue
        with open(caption_video_json, 'w', encoding = 'utf-8') as f:
            json.dump(video_caption_dict, f)
        print('Success :{}'.format(caption_video_json))


def ofa(args):
    
    """https://huggingface.co/OFA-Sys/ofa-large"""
    
    ########################################  model start #############################
    
    ckpt_dir = 'OFA-Sys/ofa-large-caption'
    # ckpt_dir = 'ofa-large-caption'
    tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
    # tokenizer = OFATokenizer.from_pretrained(ckpt_dir, use_fast=False)
    model = OFAModel.from_pretrained(ckpt_dir, use_cache=True).cuda(args.local_rank)
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

        # 好像报错我记得
        # if not osp.exists(args.exist_caption_id_list_json):

        #     command = "find {} -name '*_caption.json'".format(args.vat_root)
        #     output = subprocess.check_output(command, shell=True).decode().strip()

        #     # 将输出结果按行拆分并保存到一个列表中
        #     file_list = output.split('\n')
        #     # 将列表转换为JSON字符串
        #     json_list = json.dumps(file_list)

        #     # 将JSON字符串写入文件
        #     json_path = args.exist_caption_id_list_json
        #     with open(json_path, 'w', encoding='utf-8') as f:
        #         f.write(json_list)
        #     print('{} is saved, nums of caption file is {}'.format(json_path,len(file_list)))
        
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
                                             drop_last=True,
                                             )
    ######################################## dataset over  #############################

    ######################################## ofa caption start #############################   
    txt = " what does the image describe?"
    inputs_ids = tokenizer([txt for i in range(args.batch_size * args.caption_nums_per_video)], return_tensors="pt").input_ids

    for index, (video_ids, video_paths,  videos_frames) in enumerate(dataloader):
        bs, cap_nums, c, h, w = videos_frames.shape
        videos_frames = videos_frames.view(-1, c, h, w )
        
        # import ipdb
        # ipdb.set_trace()
        gen = model.module.generate(inputs_ids.cuda(args.local_rank), patch_images=videos_frames.cuda(args.local_rank), num_beams=5, no_repeat_ngram_size=3) 
        caption_list = tokenizer.batch_decode(gen, skip_special_tokens=True)
        ids_captions_save(args, video_ids, video_paths, caption_list)
    ######################################## ofa caption  over  #############################  
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

def glob1(path = '/apdcephfs_cq3/share_1311970/A_Youtube/coco_vat', json_path = None):
    import time
    t1 = time.time()
    import glob
    file_list = glob.glob('{}/*_caption.json'.format(path), recursive=True)
    json_list = json.dumps(file_list)

    print(f'caption.json sum is {len(json_list)}')

    # 将JSON字符串写入文件
    # json_path = 'coco_vat_exist_caption_id_list_03141026.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_list)
    print('{} is saved, nums of caption file is {}'.format(json_path,len(file_list)))
    t2 = time.time()
    print('!!!!!!!!{}s'.format(t2-t1))
    return file_list

def main(args):
    # args.rank = int(os.environ['RANK'])  # 获取当前进程号
    # args.world_size = int(os.environ['WORLD_SIZE'])
    # args.local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(args.local_rank)

    # dist.init_process_group(
    #     backend='nccl',init_method='env://',world_size=args.world_size,rank=args.rank
    # )  
    # assert torch.distributed.is_initialized()
    # dist.barrier()
    misc.init_distributed_mode(args)
    if args.rank == 0:
        print('进程组初始化完成')
        print("started")
        print("started caption_json count!")
        # glob1(json_path=args.exist_caption_id_list_json)  # 'coco_vat_exist_caption_id_list_03141026.json'
    ###########################################################3
    import time
    t1=time.time()
    ofa(args)
    t2 = time.time()
    if args.rank == 0:
        print('Time : ',t2-t1,' s')
    dist.destroy_process_group()  # 销毁进程组
    
    
def test_dataset(args):
    # command = "find {} -name '*_caption.json'".format(args.vat_root)
    # output = subprocess.check_output(command, shell=True).decode().strip()

    # # 将输出结果按行拆分并保存到一个列表中
    # file_list = output.split('\n')
    # # 将列表转换为JSON字符串
    # json_list = json.dumps(file_list)

    # # 将JSON字符串写入文件
    # json_path = args.exist_caption_id_list_json
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     f.write(json_list)
    # print('{} is saved, nums of caption file is {}'.format(json_path,len(file_list)))

    train_dataset = my_dataset(args)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    from time import time
    for i, sample in enumerate(loader):
        video_ids, video_paths,  videos_frames = sample
        
        # import ipdb
        # ipdb.set_trace()
        print(i, video_ids, video_paths, videos_frames.shape)
        



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
    args = parser.parse_args()

    main(args)

    success_file=f"part_{args.part_index}_success"
    os.system(f"touch {success_file}")
    
    """ 
    python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 29504  ofa_ddp.py \
    --train_file  "/apdcephfs_cq3/share_1311970/A_Youtube/coco_vat_890w_id_title_folderidx_merge.json"   \
    --num_workers  8   --batch_size 1    \
    --part_index 11 \
    --part_nums  10000
    
    """



