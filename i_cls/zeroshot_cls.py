
import json
import logging
import os
from training.distributed import is_master
from .zero_shot import zero_shot_eval

try:
    import wandb
except ImportError:
    wandb = None



def evaluate_i_cls(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data['i_cls'], epoch, args)
    metrics.update(zero_shot_metrics)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/i_cls/{name}", val, epoch)
        args.i_cls_output_dir = os.path.join(args.log_base_path, 'i_cls')
        os.makedirs(args.i_cls_output_dir, exist_ok=True)
        with open(os.path.join(args.i_cls_output_dir, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics
