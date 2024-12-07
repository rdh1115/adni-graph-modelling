"""Logging."""

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys

import wandb
import simplejson
import torch
import torch.distributed as dist
from iopath.common.file_io import g_pathmgr as pathmgr


def is_master_proc(multinode=False):
    """
    Determines if the current process is the master process.
    """
    if dist.is_initialized():
        if multinode:
            return dist.get_rank() % dist.get_world_size() == 0
        else:
            return dist.get_rank() % torch.cuda.device_count() == 0
    else:
        return True


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # Use 1K buffer if writing to cloud storage.
    io = pathmgr.open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io


def setup_logging(output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    if is_master_proc():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if is_master_proc():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if output_dir is not None and is_master_proc(multinode=True):
        filename = os.path.join(output_dir, "stdout.log")
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.5f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    print("json_stats: {:s}".format(json_stats))


def master_print(*args, **kwargs):
    if is_master_proc():
        print(*args, **kwargs)
    else:
        pass


def get_run_name(args):
    wandb_run_name = ''
    if 'mask_ratio' in args:
        wandb_run_name += f'mask_{args.mask_ratio}_'
    elif 'mask' in getattr(args, 'finetune', ''):
        try:
            mask_ratio = args.finetune.split('mask')[1].split('_')[1]
            wandb_run_name += f'pre_mask_{mask_ratio}_'
        except Exception as _:
            pass
    if 'gnn_mlp_' in args.model:
        wandb_run_name += f'gcn_type_{args.gcn_type}_'
    if args.dataset_name == 'ADNI':
        wandb_run_name += f'scans_{args.filter_list}_norm_{args.normalize}_petvolume_{args.include_pet_volume}_'
        if args.task == 'pred':
            wandb_run_name += f'hist_{args.n_hist}_pred_{args.n_pred}_'
        elif args.task == 'class':
            wandb_run_name += f'visits_{args.num_visits}_'
        else:
            raise ValueError(f'Invalid task: {args.task}')
    if args.test:
        if len(args.finetune.split('/')) > 1:
            wandb_run_name += args.finetune.split('/')[-2]
    else:
        wandb_run_name += f'blr_{args.blr}_wd_{args.weight_decay}_cg_{args.clip_grad}_dropout_{args.dropout}'
    print(f'logging to wandb with run name: {wandb_run_name}')
    return wandb_run_name


def setup_wandb(args, global_rank, device):
    if global_rank == 0:
        wandb_run_name = get_run_name(args)
        if not args.wandb_run_id:
            wandb_run = wandb.init(
                project=args.project_name,
                mode='offline' if args.wandb_offline else 'online',
                name=wandb_run_name,
            )
        else:
            wandb_run = wandb.init(
                project=args.project_name,
                id=args.wandb_run_id,
                resume="allow",
                mode='offline' if args.wandb_offline else 'online',
            )
        config_dict = wandb_run.config.as_dict()
        tmp = [config_dict]
        if args.distributed:
            dist.barrier()
    else:
        wandb_run = None
        tmp = [None]
        dist.barrier()
    if args.distributed:
        dist.broadcast_object_list(tmp, src=0, device=device)
    wandb_params = tmp[0]
    return wandb_run, wandb_params


def wandb_log_graph(logger, task, args, outputs, targets):
    if task == 'pred':
        columns = ['pred', 'target']
        num_steps = args.n_pred
        xs = [i for i in range(num_steps)]

        ys = [outputs.view(-1, 12, 325, 1)[0, ..., 1, 0].tolist()] + [
            targets.view(-1, 12, 325, 1)[0, ..., 1, 0].tolist()]
    logger.log(
        {
            'pred_vs_target on node 1': wandb.plot.line_series(
                ys=ys,
                xs=xs,
                keys=columns,
                title='pred_vs_target',
                xname='time',
            )
        }
    )
    return
