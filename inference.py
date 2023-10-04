import sys
from model.languagebind import LanguageBind, stack_dict
from training.params import parse_args
import torch
from data.process_image import load_and_transform_image, get_image_transform
from data.process_audio import load_and_transform_audio, get_audio_transform
from data.process_video import load_and_transform_video, get_video_transform
from data.process_depth import load_and_transform_depth, get_depth_transform
from data.process_thermal import load_and_transform_thermal, get_thermal_transform
from data.process_text import load_and_transform_text
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args.device = 'cuda:0'
    args.cache_dir = 'tokenizer_cache'
    device = torch.device(args.device)
    model = LanguageBind(args)
    ckpt = torch.load(args.languagebind_weight, map_location='cpu')
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    modality_transform = {
        'language': get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir),
        'video': get_video_transform(args),
        'audio': get_audio_transform(args),
        'depth': get_depth_transform(args),
        'thermal': get_thermal_transform(args),
        'image': get_image_transform(args),
    }

    image = ['assets/zHSOYcZblvY_resize256/0.jpg', 'assets/zlmxeeMOGVQ_resize256/0.jpg']
    # audio = ['your/audio1.wav', 'your/audio2.wav']
    video = ['assets/zHSOYcZblvY.mp4', 'assets/zlmxeeMOGVQ.mp4']
    depth = ['assets/zHSOYcZblvY_depth/0.png', 'assets/zlmxeeMOGVQ_depth/0.png']
    thermal = ['assets/zHSOYcZblvY_thermal/0.jpg', 'assets/zlmxeeMOGVQ_thermal/0.jpg']
    language = ["Training a parakeet to climb up a ladder.", 'Riding a motorcycle.']

    inputs = {
                 'image': stack_dict([load_and_transform_image(i, modality_transform['image']) for i in image], device),
                 'video': stack_dict([load_and_transform_video(i, modality_transform['video']) for i in video], device),
                 # 'audio': stack_dict([load_and_transform_audio(i, modality_transform['audio']) for i in audio], device),
                 'thermal': stack_dict([load_and_transform_thermal(i, modality_transform['thermal']) for i in thermal], device),
                 'depth': stack_dict([load_and_transform_depth(i, modality_transform['depth']) for i in depth], device),
                 'language': stack_dict([load_and_transform_text(i, modality_transform['language']) for i in language], device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    print("Video x Text: \n", torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    print("Image x Text: \n", torch.softmax(embeddings['image'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    print("Depth x Text: \n", torch.softmax(embeddings['depth'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    # print("Audio x Text: \n", torch.softmax(embeddings['audio'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    print("Thermal x Text: \n", torch.softmax(embeddings['thermal'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())