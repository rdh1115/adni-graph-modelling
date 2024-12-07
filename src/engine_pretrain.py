# --------------------------------------------------------
# References:
# mae_st: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import math
import sys
from typing import Iterable
from iopath.common.file_io import g_pathmgr as pathmgr

import torch
import utils.lr_schedule as lr_sched
import utils.misc as misc


def train_one_epoch(
        model: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        log_writer=None,
        wandb_log=None,
        args=None,
        fp32=False,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batched_data in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )
        del batched_data['y']
        batched_data = misc.prepare_batch(batched_data, device=device)
        with torch.cuda.amp.autocast(enabled=not fp32):
            loss, pred, mask = model(
                batched_data,
                mask_ratio=args.mask_ratio,
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as _:
                    pass
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(mask_ratio=args.mask_ratio)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            """
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            if log_writer is not None:
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)
            elif wandb_log is not None:
                wandb_log.log(
                    {
                        'train_loss': loss_value_reduce,
                        'lr': lr,
                        'epoch_1000': epoch_1000x
                    }
                )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
