import logging
import argparse
import os.path
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, CLIPPreTrainedModel


from model.base_model import CLIPModel
from model.process_clip import add_time_attn_block, convert_model_to_lora, set_global_value, resize_pos
from open_clip import convert_weights_to_lp
from open_clip.transformer import PatchDropout
from training.distributed import is_master


def SET_GLOBAL_VALUE(k, v):
    set_global_value(k, v)

def create_vat_model(args):

    config = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = CLIPModel(config, args.num_frames, args.add_time_attn, args.clip_type=='vl_new', args.tube_size)

    model.vision_model.patch_dropout = PatchDropout(args.force_patch_dropout)

    device = args.device
    precision = args.precision
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device)
        convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    if args.pretrained:
        try:
            args.pretrained = os.path.join(args.cache_dir, args.pretrained)
            if is_master(args):
                logging.info(f'Loading pretrained {args.model} weights ({args.pretrained}).')
            # incompatible_keys = load_checkpoint(model, pretrained, strict=False)
            ckpt = torch.load(args.pretrained, map_location='cpu')
            incompatible_keys = model.load_state_dict(ckpt, strict=False if args.add_time_attn else True)
            if is_master(args):
                logging.info(incompatible_keys)
        except Exception as e:
            if is_master(args):
                logging.info(f"Failed loading pretrained model with {e}")
    else:
        if is_master(args):
            logging.info(f"No pretrained model to load in \'{args.pretrained}\'")

    if args.add_time_attn:
        add_time_attn_block(model.vision_model.encoder, device=device)
        if is_master(args):
            logging.info(f'Convert spatial attention to time attention pretrained.')

    if args.clip_type == 'al':
        resize_pos(model.vision_model.embeddings, args)
        if is_master(args):
            logging.info(f'Resize to position embedding successfully.')

    if args.clip_type == 'vl_new':
        model.vision_model.embeddings.expand3d()

    if args.init_temp != 0:
        with torch.no_grad():
            model.logit_scale.fill_(np.log(1 / float(args.init_temp)))
        if is_master(args):
            logging.info(f'Reset logit scale to {args.init_temp} (log-scale) and trainable {args.learn_temp}.')

    if args.convert_to_lora:
        convert_model_to_lora(args, model)
        if is_master(args):
            logging.info(f"Successfuly convert model to lora style.")

    # if output_dict and hasattr(model, "output_dict"):
    #     model.output_dict = True

    return model


if __name__ == '__main__':
    MODEL_DICT = {"ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
                  "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"}
    CHECKPOINT_DICT = {"ViT-L-14": "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/pytorch_model.bin",
                       "ViT-H-14": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/pytorch_model.bin"}

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.pretrained = True
    args.model = MODEL_DICT["ViT-L-14"]
    args.pretrained = CHECKPOINT_DICT["ViT-L-14"]
    args.cache_dir = 'D:\Omni-modal-valdt-1kw'
    args.device = 'cpu'
    args.precision = None
    args.lock_text = True
    args.lock_image = True
    args.init_temp = 0
    args.force_patch_dropout = 0.5
    args.add_time_attn = True
    args.convert_to_lora = True
    args.lora_r = 16
    args.lora_alpha = 16
    args.lora_dropout = 0.0  # 0.1?
    args.num_frames = 8
    args.tube_size = 1
    args.clip_type = 'vl_new'
    args.num_mel_bins = 128
    args.target_length = 1024
    args.audio_sample_rate = 16000
    args.audio_mean = 1
    args.audio_std = 1
    args.rank = 0

    # SET_GLOBAL_VALUE('PATCH_DROPOUT', args.force_patch_dropout)
    # SET_GLOBAL_VALUE('NUM_FRAMES', args.num_frames)

    model = create_vat_model(args)


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


    model_structure(model)
    # model_structure(model.vision_model)
    # model_structure(model.text_model)


    # model.lock_image_tower(unlocked_groups=1)
    # model.lock_text_tower(unlocked_layers=0)
    # model.unlock_time_attn()

    if args.lock_image:
        # if args.clip_type == 'al' or args.clip_type == 'dl':
        #     for param in model.vision_model.embeddings.parameters():
        #         param.requires_grad = True
        #     for param in model.vision_model.pre_layrnorm.parameters():
        #         param.requires_grad = True
        # else:
        for param in model.vision_model.embeddings.parameters():
            param.requires_grad = False
        for param in model.vision_model.pre_layrnorm.parameters():
            param.requires_grad = False
    for param in model.vision_model.embeddings.position_embedding.parameters():
        param.requires_grad = False
    model.vision_model.embeddings.class_embedding.requires_grad = True


    if args.lock_text:
        for param in model.text_model.parameters():
            param.requires_grad = False
        for param in model.text_projection.parameters():
            param.requires_grad = False


    for n, p in model.named_parameters():
        # if p.requires_grad:
        print(n, '--->', p.requires_grad)
    b, c, t, h, w = 2, 3, args.num_frames, 224, 224
    x = torch.randn(b, c, t, h, w)
    y = model(image=x)
    print()