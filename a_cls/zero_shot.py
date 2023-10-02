import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from .precision import get_autocast
from .stats import calculate_stats, d_prime
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


def validate(audio_model, classifier, val_loader, args, epoch):
    epoch = epoch - 1 ########################
    # switch to evaluate mode
    audio_model.eval()
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(tqdm(val_loader)):
            audio_input = audio_input.to(device=args.device, dtype=input_dtype)

            # compute output
            with autocast():
                # predict
                output = audio_model(image=audio_input)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier
            audio_output = logits

            # audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(args.device)
            loss = nn.CrossEntropyLoss()(audio_output, torch.argmax(labels.long(), dim=1))
            A_loss.append(loss.to('cpu').detach())

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        # save the prediction here
        args.a_cls_output_dir = os.path.join(args.log_base_path, f'a_cls/{args.val_a_cls_data.lower()}')
        os.makedirs(args.a_cls_output_dir, exist_ok=True)
        if os.path.exists(args.a_cls_output_dir + '/predictions') == False:
            os.mkdir(args.a_cls_output_dir + '/predictions')
            np.savetxt(args.a_cls_output_dir + '/predictions/target.csv', target, delimiter=',')
        np.savetxt(args.a_cls_output_dir + '/predictions/predictions_' + str(epoch) + '.csv', audio_output,
                   delimiter=',')

    valid_loss = loss
    main_metrics = 'mAP'
    metrics = {}

    if args.do_train:
        # ensemble results
        cum_stats = validate_ensemble(args, epoch)
        cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        cum_acc = cum_stats[0]['acc']

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    acc = stats[0]['acc']

    middle_ps = [stat['precisions'][int(len(stat['precisions']) / 2)] for stat in stats]
    middle_rs = [stat['recalls'][int(len(stat['recalls']) / 2)] for stat in stats]
    average_precision = np.mean(middle_ps)
    average_recall = np.mean(middle_rs)

    if main_metrics == 'mAP':
        logging.info("mAP: {:.6f}".format(mAP))
    else:
        logging.info("acc: {:.6f}".format(acc))
    logging.info("AUC: {:.6f}".format(mAUC))
    logging.info("Avg Precision: {:.6f}".format(average_precision))
    logging.info("Avg Recall: {:.6f}".format(average_recall))
    logging.info("d_prime: {:.6f}".format(d_prime(mAUC)))
    logging.info("valid_loss: {:.6f}".format(valid_loss))

    if args.do_train:
        logging.info("cum_mAP: {:.6f}".format(cum_mAP))
        logging.info("cum_mAUC: {:.6f}".format(cum_mAUC))

    if main_metrics == 'mAP':
        metrics['mAP'] = float(mAP)
    else:
        metrics['acc'] = float(acc)

    metrics['mAUC'] = float(mAUC)
    metrics['average_precision'] = float(average_precision)
    metrics['average_recall'] = float(average_recall)
    metrics['d_prime_mAUC'] = float(d_prime(mAUC))
    metrics['valid_loss'] = float(valid_loss)

    if args.do_train:
        metrics['cum_mAP'] = float(cum_mAP)
        metrics['cum_mAUC'] = float(cum_mAUC)

    return metrics


def validate_ensemble(args, epoch):
    exp_dir = args.a_cls_output_dir
    target = np.loadtxt(exp_dir + '/predictions/target.csv', delimiter=',')
    if epoch == 0:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_0.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir + '/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir + '/predictions/predictions_' + str(epoch - 1) + '.csv')

    cum_predictions = cum_predictions / (epoch + 1)
    np.savetxt(exp_dir + '/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats









def zero_shot_eval(model, data, epoch, args):
    temp_val_a_cls_data = args.val_a_cls_data
    args.val_a_cls_data = list(data.keys())
    assert len(args.val_a_cls_data) == 1
    args.val_a_cls_data = args.val_a_cls_data[0]

    if args.val_a_cls_data not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info(f'Starting zero-shot {args.val_a_cls_data.upper()}.')

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        tokenizer = get_tokenizer(HF_HUB_PREFIX+args.model, cache_dir=args.cache_dir)
        # tokenizer = get_tokenizer("ViT-L-14")
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=CLASSNAMES[args.val_a_cls_data],
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if args.val_a_cls_data.lower() == 'audioset':
        if args.val_a_cls_data in data:
            stats = validate(model, classifier, data[args.val_a_cls_data].dataloader, args, epoch)
            results.update(stats)
    else:
        if args.val_a_cls_data in data:
            top1, top5 = run(model, classifier, data[args.val_a_cls_data].dataloader, args)
            results[f'{args.val_a_cls_data}-zeroshot-val-top1'] = top1
            results[f'{args.val_a_cls_data}-zeroshot-val-top5'] = top5

    logging.info(f'Finished zero-shot {args.val_a_cls_data.upper()}.')

    args.val_a_cls_data = temp_val_a_cls_data
    return results











