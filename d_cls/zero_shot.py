import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from .precision import get_autocast
from .zero_shot_classifier import build_zero_shot_classifier
from .zero_shot_metadata import CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            images = images.unsqueeze(2)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args):
    temp_val_d_cls_data = args.val_d_cls_data
    args.val_d_cls_data = list(data.keys())
    assert len(args.val_d_cls_data) == 1
    args.val_d_cls_data = args.val_d_cls_data[0]

    if args.val_d_cls_data not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info(f'Starting zero-shot {args.val_d_cls_data.upper()}.')

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        tokenizer = get_tokenizer(HF_HUB_PREFIX+args.model, cache_dir=args.cache_dir)
        # tokenizer = get_tokenizer("ViT-L-14")
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=CLASSNAMES[args.val_d_cls_data],
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if args.val_d_cls_data in data:
        top1, top5 = run(model, classifier, data[args.val_d_cls_data].dataloader, args)
        results[f'{args.val_d_cls_data}-zeroshot-val-top1'] = top1
        results[f'{args.val_d_cls_data}-zeroshot-val-top5'] = top5

    logging.info(f'Finished zero-shot {args.val_d_cls_data.upper()}.')

    args.val_d_cls_data = temp_val_d_cls_data
    return results
