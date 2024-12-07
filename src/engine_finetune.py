# --------------------------------------------------------
# References:
# mae_st: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch

import utils.lr_schedule as lr_sched
import utils.misc as misc
from utils.log import wandb_log_graph
from timm.utils import accuracy


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
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
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    task = args.task
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

        batched_data = misc.prepare_batch(batched_data, device=device)
        scaler = None if 'scaler' not in batched_data else batched_data['scaler']

        samples, targets, target_shape = misc.get_samples_targets(batched_data, task, device, args)
        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(batched_data)
            if scaler and task == 'pred':
                outputs = scaler.inverse_transform(outputs, args)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            create_graph=False,
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

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

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
                log_writer.add_scalar("lr", max_lr, epoch_1000x)
            elif wandb_log is not None:
                wandb_log.log(
                    {
                        'train_loss': loss_value_reduce,
                        'lr': max_lr,
                        'epoch_1000': epoch_1000x
                    }
                )
                if args.wandb_watch:
                    wandb_log_graph(wandb_log, args, outputs, targets)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, criterion, device, args):
    task = args.task

    if criterion is None:
        if task == 'class':
            criterion = torch.nn.CrossEntropyLoss()
        elif task == 'pred':
            criterion = torch.nn.L1Loss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation atlas
    model.eval()

    scaler = None
    for batched_data in metric_logger.log_every(data_loader, 10, header):
        batched_data = misc.prepare_batch(batched_data, device=device)
        scaler = None if 'scaler' not in batched_data else batched_data['scaler']

        samples, targets, target_shape = misc.get_samples_targets(batched_data, task, device, args)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(batched_data)
            if scaler and task == 'pred':
                outputs = scaler.inverse_transform(outputs, args)
                targets = scaler.inverse_transform(targets, args)
            loss = criterion(outputs, targets)

        metric_logger.update(loss=loss.item())
        batch_size = samples.shape[0]

        if task == 'class':
            k = 5 if args.pred_num_classes > 3 else 2
            acc1, acck = accuracy(outputs, targets, topk=(1, k))

            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters[f"acc{k}"].update(acck.item(), n=batch_size)
        elif task == 'pred':
            metrics = misc.forecasting_acc(
                outputs,
                targets,
                target_shape=None if args.n_hist == args.n_pred or args.n_pred > args.n_hist
                else target_shape
            )

            metric_logger.meters['mae'].update(metrics['MAE'].item(), n=batch_size)
            metric_logger.meters['rmse'].update(metrics['RMSE'].item(), n=batch_size)
            metric_logger.meters['mape'].update(metrics['MAPE'].item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    if task == 'class':
        k = 5 if args.pred_num_classes > 3 else 2
        acck_global_avg = metric_logger.meters[f"acc{k}"].global_avg
        print(
            f"* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@{k} {acck_global_avg:.3f} "
            f"loss {metric_logger.loss.global_avg:.3f}"
        )
    elif task == 'pred':
        print(
            "* MAE {mae.global_avg:.3f} RMSE {rmse.global_avg:.3f} MAPE {mape.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                mae=metric_logger.mae, rmse=metric_logger.rmse, mape=metric_logger.mape, losses=metric_logger.loss
            )
        )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
