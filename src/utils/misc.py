# --------------------------------------------------------
# References:
# mae_st: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import argparse
import builtins
import datetime
import re
import os
import time
from collections import defaultdict, deque

import numpy as np

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from pathlib import Path

import psutil
import socket

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import inf
from timm.models.layers import trunc_normal_

from iopath.common.file_io import g_pathmgr as pathmgr

from src.utils import log
from src.utils.log import master_print as print
from src.utils.pos_embed import interpolate_pos_embed
from src.data.utils import DX_DICT, get_target_brain_arr_indices

logger = log.get_logger(__name__)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )

                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def forecasting_acc(output, target, target_shape=None):
    """
    calculate 3 time-series prediction metric:
    MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error)
    mask out values of 0 from target to prevent MAPE from numerical error

    :param output: prediction tensor from model
    :param target: ground truth tensor
    :param target_shape: tuple of the shape of target tensor
    :return:
    """
    if target_shape is not None:
        # if target future time steps is 3, but output future time step is 12,
        # then pad future time step, and only evaluate 3 time steps from output
        N, P, V, D = target_shape
        output = output.view(N, -1, V, D)
        T = output.shape[1]
        assert T >= P, 'cannot predict less time steps during inference'
        output = output[:, :P, ...].view(N, -1, D)
        assert output.shape == target.shape, f'output shape {output.shape} must match target shape {target.shape}'

    metrics = ['MAE', 'RMSE', 'MAPE']
    metrics = dict.fromkeys(metrics, None)

    # mask out node values that are zero during evaluation
    # this avoids MAPE numerical error
    mask = (target != 0)

    mae = (output - target).abs()
    mae = mae * mask
    mae = torch.where(torch.isnan(mae), torch.zeros_like(mae), mae)
    metrics['MAE'] = mae.mean()

    rmse = (output - target) ** 2
    rmse = rmse * mask
    rmse = torch.where(torch.isnan(rmse), torch.zeros_like(rmse), rmse)
    metrics['RMSE'] = torch.sqrt(rmse.mean())

    mape = ((output - target).abs() / target.abs())
    mape = mape * mask
    mape = torch.where(torch.isnan(mape), torch.zeros_like(mape), mape)
    metrics['MAPE'] = mape.mean()
    return metrics


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, path):
    if is_main_process():
        print(f"save path {path}")
        with pathmgr.open(path, "wb") as f:
            torch.save(state, f)


def init_distributed_mode(args):
    ngpus_per_node = torch.cuda.device_count()
    print('gpus_per_node: ', ngpus_per_node)
    if args.no_env:
        pass
    elif ngpus_per_node == 0:
        print("Not using distributed atlas")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return
    elif args.torch_run:
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        # args.dist_url = "tcp://%s:%s" % (
        #     os.environ["MASTER_ADDR"],
        #     3456,
        # )
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "SLURM_LOCALID" in os.environ and "SLURM_NODEID" in os.environ:
        args.gpu = int(os.environ.get("SLURM_LOCALID"))
        args.rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + args.gpu
        assert args.rank == int(os.environ.get("SLURM_PROCID"))
        # find the master node from node list
        master_node_id = min(map(int, re.findall(r"\d+", os.environ.get("SLURM_JOB_NODELIST"))))
        if not args.dist_url:
            master_addr = os.environ.get('MASTER_ADDR')
            if not master_addr:
                hostname = socket.gethostname()
                master_addr = re.sub(r"\d+", str(master_node_id), hostname)
            args.dist_url = f"tcp://{master_addr}:3456"
        if not args.world_size:
            args.world_size = int(os.environ.get("SLURM_JOB_NUM_NODES")) * int(os.environ.get("SLURM_NTASKS_PER_NODE"))
        assert args.world_size == int(os.environ.get("SLURM_NTASKS")), \
            (f'world_size: {args.world_size}, SLURM_NTASKS: {os.environ.get("SLURM_NTASKS")}, '
             f'SLURM_NTASKS_PER_NODE: {os.environ.get("SLURM_NTASKS_PER_NODE")}, '
             f'SLURM_JOB_NUM_NODES: {os.environ.get("SLURM_JOB_NUM_NODES")}')
    else:
        print("Not using distributed atlas")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    if args.dist_backend is None:
        args.dist_backend = "nccl"

    print('From Rank: {}, ==> Initializing Process Group...'.format(args.rank))
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        # flush=True,
    )

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    print("process group ready!")
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, fp32=False):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not fp32)

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            parameters=None,
            create_graph=False,
            update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    save_dir = args.output_dir
    checkpoint_path = "{}/checkpoint-{:05d}.pth".format(save_dir, epoch)
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }

    save_on_master(to_save, checkpoint_path)
    if args.distributed:
        dist.barrier()
    return checkpoint_path


def get_last_checkpoint(args):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = args.output_dir
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        print("No checkpoints found in '{}'.".format(d))
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names)[-1]
        checkpoint = os.path.join(d, name)
        print(f"Found checkpoint at {checkpoint}")
        return checkpoint


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    """
    Load the model from checkpoint.
    if args.output_dir is not empty, load the model from the last checkpoint in args.output_dir, the fp is args.resume

    this is used for both pre-training and finetuning,
    so make sure output_dir is empty if finetuning from pretrained checkpoint

    if args.output_dir is empty, and args.finetune is provided (pretrained checkpoint fp),
    load the model for finetuning.

    all trained models are then saved at args.output_dir

    :param args:
    :param model_without_ddp:
    :param optimizer:
    :param loss_scaler:
    :return:
    """
    if not args.resume:
        args.resume = get_last_checkpoint(args)

    checkpoint = None
    finetune_from_pretrain = False
    if args.resume:
        # Load checkpoint from URL or local path
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            with pathmgr.open(args.resume, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
    elif getattr(args, 'finetune', None):
        # If finetuning and pretrained checkpoint provided
        with pathmgr.open(args.finetune, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        finetune_from_pretrain = True
    else:
        # No valid checkpoint found, proceeding to train from scratch
        output_dir = args.output_dir
        print(f"No checkpoint found.\nTraining from scratch in directory: {output_dir}")
        return

    if checkpoint:
        args.start_epoch = checkpoint["epoch"] + 1

        if "model" in checkpoint.keys():
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint["model_state"]

        if finetune_from_pretrain:
            state_dict = model_without_ddp.state_dict()
            for k in ["head.weight", "head.bias"]:
                if (
                        k in checkpoint_model and k in state_dict
                        and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            msg = model_without_ddp.load_state_dict(checkpoint_model, strict=False)
            if 'head.weight' in model_without_ddp.state_dict():
                trunc_normal_(model_without_ddp.head.weight, std=2e-5)
            if 'head.bias' in model_without_ddp.state_dict():
                nn.init.constant_(model_without_ddp.head.bias, 0)
            if 'norm.weight' in model_without_ddp.state_dict():
                nn.init.constant_(model_without_ddp.norm.weight, 1)
            if 'norm.bias' in model_without_ddp.state_dict():
                nn.init.constant_(model_without_ddp.norm.bias, 0)
            print("Finetuning from pretrain checkpoint %s" % args.finetune)
        else:
            msg = model_without_ddp.load_state_dict(checkpoint_model)
            print("Resume checkpoint %s" % args.resume)
            if (
                    "optimizer" in checkpoint
                    and "epoch" in checkpoint
                    and not (hasattr(args, "eval") and args.eval)
            ):
                optimizer.load_state_dict(checkpoint["optimizer"])
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])
                print("Also resume optim & sched checkpoint")
        print(msg)
    return


def load_finetune(args, model):
    """
    used for testing only
    :param args:
    :param model:
    :return:
    """
    with pathmgr.open(args.finetune, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    if "model" in checkpoint.keys():
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint["model_state"]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    return model


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.barrier()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
                (not bias_wd)
                and len(param.shape) == 1
                or name.endswith(".bias")
                or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def get_updates(data_sample, args: argparse.Namespace, wandb_params: dict):
    if wandb_params:
        vals = list()
        for param, v in wandb_params.items():
            setattr(args, param, v)
            if isinstance(v, Iterable):
                vals += [b for b in v]
            else:
                vals.append(v)

    if not args.test:
        eff_batch_size = args.batch_size * args.accum_iter * get_world_size()
        args.eff_batch_size = eff_batch_size
        args.lr = args.blr * eff_batch_size / 256 if eff_batch_size >= 256 else args.blr

    if args.static_graph:
        args.node_feature_dim = data_sample['x'].shape[-1]
        args.num_nodes = data_sample['adj'].shape[0]
        args.num_edges = len(data_sample['edge_attr'])

        # account for data.utils.collator changes
        args.num_spatial = torch.max(data_sample['spatial_pos']).item() + 1
        args.num_in_degree = torch.max(data_sample['in_degree']).item() + 1
        args.num_out_degree = torch.max(data_sample['out_degree']).item() + 1
    else:
        raise NotImplementedError()

    return


def get_samples_targets(batch, task, device, args=None):
    # scaler = batch['scaler']
    samples, targets = batch['x'], batch['y']
    target_shape = targets.shape
    # targets = scaler.inverse_transform(targets) if scaler else targets
    if task == 'pred':
        N, P, V, D = target_shape
        pred_dim_idx = get_target_brain_arr_indices(D, args.filter_list, args.include_pet_volume)
        targets = targets[..., pred_dim_idx]
        D = len(pred_dim_idx)
        target_shape = targets.shape
        targets = targets.contiguous().view(N, -1, D)
    elif task == 'class':
        if not args:
            dataset_name = 'ADNI'
            pred_num_classes = 3
            pred_per_time_step = False
        else:
            dataset_name = args.dataset_name
            pred_num_classes = args.pred_num_classes
            pred_per_time_step = args.pred_per_time_step

        features = batch['add_features']
        if 'ADNI' == dataset_name:
            if pred_num_classes == 3 and not pred_per_time_step:
                targets = targets.flatten()
            elif pred_num_classes == 9 and pred_per_time_step:
                feature_pred = 'DX_CHANGE'
                targets = torch.stack(
                    [d[feature_pred].long() for d in features]
                ).to(device, non_blocking=True)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    return samples, targets, target_shape


def prepare_batch(batched_data, device):
    for k, v in batched_data.items():
        if k == 'scaler':
            if v is not None:
                v = v.to_device(device)
                batched_data[k] = v
        elif v is not None and k != 'add_features':
            batched_data[k] = v.to(device, non_blocking=True)
    return batched_data
