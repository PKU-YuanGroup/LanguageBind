
import gradio as gr
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
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX






class LanguageBind(nn.Module):
    def __init__(self, args, no_temp=False):
        super(LanguageBind, self).__init__()
        self.no_temp = no_temp
        MODEL_DICT = {"ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
                      "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"}
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
            elif c == 'vl':
                args.lora_r = 64
                args.add_time_attn = True
                model = create_vat_model(args)
                args.add_time_attn = False
                args.lora_r = 2
            elif c == 'al':
                args.lora_r = 8
                model = create_vat_model(args)
                args.lora_r = 2
            else:
                model = create_vat_model(args)
            '''
            state_dict = torch.load(f'model_zoo/{c}.pt', map_location='cpu')
            if state_dict.get('state_dict', None) is not None:
                state_dict = state_dict['state_dict']
            if next(iter(state_dict.items()))[0].startswith('module'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(f'load {c}, {msg}')
            '''
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
            if not self.no_temp:
                if key != 'language':
                    value = value * self.modality_scale[key].exp()
            outputs[key] = value
        return outputs



def stack_dict(x, device):
    if len(x) == 0:
        return None
    out_dict = {}
    keys = list(x[0].keys())
    for key in keys:
        out_dict[key] = torch.stack([i[key] for i in x]).to(device)
    return out_dict