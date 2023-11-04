import logging
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from training.distributed import is_master
from v_cls.zero_shot_classifier import build_zero_shot_classifier
from v_cls.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, CLASSNAMES

from training.precision import get_autocast




def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    file = os.path.join(args.output_dir, str(args.rank) + '.txt')
    final_result = []
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for batch in tqdm(dataloader, unit_scale=args.batch_size):
            images = batch[0]
            target = batch[1]
            ids = batch[2]
            chunk_nb = batch[3]
            split_nb = batch[4]
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier
            output = logits
            # print(output.shape)
            for i in range(output.size(0)):
                string = "{} {} {} {} {}\n".format(
                    ids[i], str(output.data[i].cpu().numpy().tolist()),
                    str(int(target[i].cpu().numpy())),
                    str(int(chunk_nb[i].cpu().numpy())),
                    str(int(split_nb[i].cpu().numpy())))
                final_result.append(string)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(top1, top5))
        for line in final_result:
            f.write(line)

    return top1, top5


def zero_shot_eval(model, dataloader, epoch, args):
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module
    if is_master(args):
        logging.info(f'Starting zero-shot {args.val_v_cls_data[0].upper()}')
        logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        tokenizer = get_tokenizer(HF_HUB_PREFIX+args.model, cache_dir=args.cache_dir)
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=CLASSNAMES[args.val_v_cls_data[0]],
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )


    if is_master(args):
        logging.info('Using classifier')
    # results = {}
    run(model, classifier, dataloader, args)
    # results['kinetics400-zeroshot-val-top1'] = top1
    # results['kinetics400-zeroshot-val-top5'] = top5

    if is_master(args):
        logging.info(f'Finished zero-shot {args.val_v_cls_data[0].upper()}.')

    # return results
