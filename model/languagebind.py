import argparse

import numpy as np
import torch
from torch import nn

from data.process_image import load_and_transform_image, get_image_transform
from main import SET_GLOBAL_VALUE
from model.build_model import create_vat_model
from data.process_audio import load_and_transform_audio, get_audio_transform
from data.process_video import load_and_transform_video, get_video_transform
from data.process_depth import load_and_transform_depth, get_depth_transform
from data.process_thermal import load_and_transform_thermal, get_thermal_transform
from data.process_text import load_and_transform_text
from model.process_clip import resize_pos
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX

'''方法1，自定义函数 参考自 https://blog.csdn.net/qq_33757398/article/details/109210240'''


def model_structure(model):
    blank = ' '
    print('-' * 150)
    print('|' + ' ' * 44 + 'weight name' + ' ' * 45 + '|' \
          + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 150)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 100:
            key = key + (100 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 150)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 150)

class LanguageBind(nn.Module):
    def __init__(self, args):
        super(LanguageBind, self).__init__()
        temp_clip_type = args.clip_type
        self.modality_encoder = {}
        self.modality_proj = {}
        self.modality_scale = {}
        for c in temp_clip_type:
            args.clip_type = c
            if c == 'il':
                args.convert_to_lora = False
                model = create_vat_model(args)
                args.convert_to_lora = True
            else:
                model = create_vat_model(args)
            # state_dict = torch.load(f'D:\Omni-modal-valdt-9.1\model_zoo/{c}.pt', map_location='cpu')
            # if state_dict.get('state_dict', None) is not None:
            #     state_dict = state_dict['state_dict']
            # if next(iter(state_dict.items()))[0].startswith('module'):
            #     state_dict = {k[7:]: v for k, v in state_dict.items()}
            # msg = model.load_state_dict(state_dict, strict=False)
            # print(f'load {c}, {msg}')
            if c == 'vl':
                self.modality_encoder['video'] = model.vision_model
                self.modality_proj['video'] = model.visual_projection
                self.modality_scale['video'] = model.logit_scale
            elif c == 'al':
                self.modality_encoder['audio'] = model.vision_model
                self.modality_proj['audio'] = model.visual_projection
                self.modality_scale['audio'] = model.logit_scale
            elif c == 'dl':
                self.modality_encoder['depth'] = model.vision_model
                self.modality_proj['depth'] = model.visual_projection
                self.modality_scale['depth'] = model.logit_scale
            elif c == 'tl':
                self.modality_encoder['thermal'] = model.vision_model
                self.modality_proj['thermal'] = model.visual_projection
                self.modality_scale['thermal'] = model.logit_scale
            elif c == 'il':
                self.modality_encoder['image'] = model.vision_model
                self.modality_proj['image'] = model.visual_projection
                self.modality_scale['image'] = model.logit_scale
            else:
                raise NameError(f'No clip_type of {c}')
        self.modality_encoder['language'] = model.text_model
        self.modality_proj['language'] = model.text_projection

        self.modality_encoder = nn.ModuleDict(self.modality_encoder)
        self.modality_proj = nn.ModuleDict(self.modality_proj)

    def forward(self, inputs):
        outputs = {}
        for key, value in inputs.items():
            value = self.modality_encoder[key](**value)[1]
            value = self.modality_proj[key](value)
            value = value / value.norm(p=2, dim=-1, keepdim=True)
            if key != 'language':
                value = value * self.modality_scale[key].exp()
            outputs[key] = value
        return outputs

def stack_dict(x, device):
    out_dict = {}
    keys = list(x[0].keys())
    for key in keys:
        out_dict[key] = torch.stack([i[key] for i in x]).to(device)
    return out_dict

if __name__ == '__main__':
    MODEL_DICT = {"ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
                  "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"}
    CHECKPOINT_DICT = {"ViT-L-14": "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/pytorch_model.bin",
                       "ViT-H-14": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/pytorch_model.bin"}

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.pretrained = False
    args.model = MODEL_DICT["ViT-L-14"]
    args.cache_dir = 'D:/Omni-modal-valdt-audio'
    args.video_decode_backend = 'decord'
    # args.device = 'cpu'
    args.device = 'cuda:0'
    device = torch.device(args.device)
    args.precision = None
    args.init_temp = 0
    args.force_patch_dropout = 0.0
    args.add_time_attn = False
    args.convert_to_lora = True
    args.lora_r = 2
    args.lora_alpha = 16
    args.lora_dropout = 0.0  # 0.1?
    args.num_frames = 8
    args.clip_type = 'vl'
    args.num_mel_bins = 1008
    args.target_length = 112
    args.audio_sample_rate = 16000
    args.audio_mean = 4.5689974
    args.audio_std = -4.2677393
    args.max_depth = 10
    args.image_size = 224
    args.rank = 0
    SET_GLOBAL_VALUE('PATCH_DROPOUT', args.force_patch_dropout)
    SET_GLOBAL_VALUE('NUM_FRAMES', args.num_frames)
    args.clip_type = ['il', 'vl', 'al', 'dl', 'tl']

    modality_transform = {
        'language': get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir),
        'video': get_video_transform(args),
        'audio': get_audio_transform(args),
        'depth': get_depth_transform(args),
        'thermal': get_thermal_transform(args),
        'image': get_image_transform(args),
    }
    # video = [
    #      r'D:\ImageBind-main\lb_test/zHSOYcZblvY.mp4',
    #      r'D:\ImageBind-main\lb_test/zlmxeeMOGVQ.mp4',
    #      r'D:\ImageBind-main\lb_test/eBPnfKjGig4.mp4',
    #      r'D:\ImageBind-main\lb_test/hQynksVwpWc.mp4',
    #      r'D:\ImageBind-main\lb_test/zwKkGRoJkvs.mp4',
    #      r'D:\ImageBind-main\lb_test/zSvb6seyeEs.mp4',
    #      r'D:\ImageBind-main\lb_test/zeImJp3Dq28.mp4',
    #      r'D:\ImageBind-main\lb_test/zamx1xZa8Ak.mp4',
    #      r'D:\ImageBind-main\lb_test/dVC8Dl0xCKg.mp4',
    # ]
    # audio = [
    #     r"D:\Omni-modal-valdt-1kw\gpt/0.mp3",
    #     r"D:\Omni-modal-valdt-1kw\gpt/1.mp3",
    #     r"D:\Omni-modal-valdt-1kw\gpt/2.mp3",
    #     r"D:\Omni-modal-valdt-1kw\gpt/3.mp3",
    #     r"D:\Omni-modal-valdt-1kw\gpt/4.mp3",
    #     r"D:\Omni-modal-valdt-1kw\gpt/5.mp3",
    #     r"D:\Omni-modal-valdt-1kw\gpt/6.mp3",
    #     r"D:\Omni-modal-valdt-1kw\gpt/7.mp3",
    #     r"D:\Omni-modal-valdt-1kw\gpt/8.mp3",
    # ]
    thermal = []
    depth = []
    # language = [
    #     "a parrot perched atop various objects, including a cardboard box, a bird cage, and a wooden keyboard, while a shark is unexpectedly discovered inside a cardboard box with a comb on top. The scene relates to the title of 'Training Your Parakeet Climbing the Ladder.'",
    #     'a man riding a green motorcycle down a city street, while wearing a gorilla suit, and losing his helmet, resembling Darth Vader.',
    #     'the awe-inspiring beauty of Switzerland with a city covered in snow, featuring a snowy path through a park adorned with trees and lanterns.',
    #     'a lion climbing a tree in an attempt to catch a monkey, as described in the title. Other scenes include a group of people standing on a tree branch, a monkey on a tree branch near a body of water, a couple of birds perched on a tree, a monkey sitting on a tree branch, a group of monkeys on a tree, a monkey climbing a tree with another monkey on top, and a polar bear jumping off a tree branch.',
    #     "various scenes, including a turtle next to a toy train on the floor, a small tree with red leaves placed on a table, and a toy train on a track situated on a table. There is also a white train positioned on top of a green table, a toy train on a track with a tree in the background, and a person standing in front of a table with a toy microwave on top of it. The title of the video is 'Amazing Technology: Train Is Swimming in the Air.'",
    #     "Ni Ki, accompanied by his model, walking down a sidewalk, with scenes showing a man in a suit and a woman in a black coat standing in front of a building. The title of the video is 'Ni Ki and His Model Walk.'",
    #     "the results of using Dollar Tree rollers on fine hair, with scenes featuring a girl with pink hair rollers standing next to another girl, a woman in a blue shirt and black pants standing next to a girl in a red shirt, and a woman with glasses holding a baby. Additionally, there are glimpses of a woman cutting her hair with a pink comb, a girl with a pink comb in her hair, a girl with glasses and a blue shirt casually brushing her hair, and a woman sitting in the back seat of a car.",
    #     "a man seated in a chair next to a brick wall, wearing a hat and smiling. Another scene shows the man sitting on top of a couch with the same brick wall backdrop. Additionally, there is an image of a yellow door with the inscription 'The Door to Ireland' on it. The title of the video is 'When You Realize What Noor He Is Talking About.'",
    #     "a 1963 Ford Thunderbird with rare factory options, featuring a white car with a red interior seat placed in the grass. There are close-up shots of the car engine with the hood open, emphasizing its details. Additionally, there is an image of the car with the words '85 T Bird Factory Operators' on its side, along with a car with an open hood situated on a grass-covered field."
    # ]

    language = ["A dog.", "A car", "A bird"]
    image = ["D:\ImageBind-main/.assets/dog_image.jpg", "D:\ImageBind-main/.assets/car_image.jpg", "D:\ImageBind-main/.assets/bird_image.jpg"]
    audio = ["D:\ImageBind-main/.assets/dog_audio.wav", "D:\ImageBind-main/.assets/car_audio.wav", "D:\ImageBind-main/.assets/bird_audio.wav"]

    inputs = {
                 'image': stack_dict([load_and_transform_image(i, modality_transform['image']) for i in image], device),
                 # 'video': stack_dict([load_and_transform_video(i, modality_transform['video']) for i in video], device),
                 'audio': stack_dict([load_and_transform_audio(i, modality_transform['audio']) for i in audio], device),
                 # 'thermal': stack_dict([load_and_transform_thermal(i, modality_transform['thermal']) for i in thermal], device),
                 # 'depth': stack_dict([load_and_transform_depth(i, modality_transform['depth']) for i in depth], device),
                 'language': stack_dict([load_and_transform_text(i, modality_transform['language']) for i in language], device)
    }

    model = LanguageBind(args).to(device)
    model.eval()

    # model_structure(model)

    with torch.no_grad():
        embeddings = model(inputs)
    torch.set_printoptions(precision=2)
    # print(
    #     "Video x Text: \n",
    #     np.around(torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy(), decimals=2),
    # )
    print(
        "Audio x Text: \n",
        np.around(torch.softmax(embeddings['audio'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy(), decimals=2),
    )
    print(
        "Image x Text: \n",
        np.around(torch.softmax(embeddings['image'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy(), decimals=2),
    )
    print(
        "Image x Audio: \n",
        np.around(torch.softmax(embeddings['image'] @ embeddings['audio'].T, dim=-1).detach().cpu().numpy(), decimals=2),
    )
    # print(
    #     "Video x Audio: \n",
    #     np.around(torch.softmax(embeddings['video'] @ embeddings['audio'].T, dim=-1).detach().cpu().numpy(), decimals=2),
    # )
