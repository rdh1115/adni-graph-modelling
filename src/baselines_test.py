# --------------------------------------------------------
# References:
# mae_st: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import json
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.distributed as dist

from iopath.common.file_io import g_pathmgr as pathmgr
import wandb

import baselines
from engine_test import test
from data.get_dataset import get_dataset
from data.utils import collator
from utils.log import master_print as print
from utils.log import get_run_name
from utils.misc import get_updates
import utils.misc as misc
from utils.model_parser import GMAEParser


def get_args_parser():
    parser = GMAEParser(
        "baseline testing for classification or prediction", add_help=False
    )

    # < model init configs
    # Model parameters
    parser.add_argument(
        "--model",
        default="graph_pred_big",
        type=str,
        metavar="MODEL",
        help="Name of model to test",
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float, default=0.1, metavar="D",
        help="classification MLP dropout probability"
    )
    parser.add_argument(
        '--gcn_type',
        default='gat',
        help='type of graph convolution for classification'
    )
    # >

    # finetune task related args
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        '--old_config',
        action="store_true",
        default=False,
        help="flag to fix some implementation differences"
    )
    parser.add_argument(
        '--max_pooling',
        action='store_true',
        default=False,
        help='if --max_pooling is specified, use average pooling'
    )
    return parser


def main(args):
    if not dist.is_initialized():  # avoid initializing twice during sweep
        misc.init_distributed_mode(args)
    if not args.project_name:
        args.project_name = f'baseline_test_{args.task}_{args.dataset_name}_{args.model}'

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available.')
        args.device = 'cpu'
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    args.test = True
    task = args.task
    if task == 'class':
        args.n_pred = 0
    if args.n_hist > 1:
        args.n_hist = 1
        print('changed --n_hist to 1 for better biological interpretation')
    args.accum_iter = 1  # no gradient accumulation during testing

    # start logging
    global_rank = misc.get_rank() if args.distributed else 0
    if args.log_dir:
        if global_rank == 0:
            try:
                pathmgr.mkdirs(args.log_dir)
            except Exception as _:
                pass
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None
    else:
        if global_rank == 0:
            wandb_run_name = get_run_name(args)
            wandb_run = wandb.init(
                project=args.project_name,
                mode='offline' if args.wandb_offline else 'online',
                name=wandb_run_name,
            )
            if args.distributed:
                dist.barrier()
        else:
            wandb_run = None
            dist.barrier()
        log_writer = None

    # get datasets
    dataset_dict = get_dataset(
        mode="test",
        data_dir=args.path_to_data_dir,
        dataset_name=args.dataset_name,
        n_hist=args.n_hist,
        n_pred=args.n_pred,
        split_ratio=tuple(args.split_ratio),
        graph_token=args.graph_token,
        seed=seed,
        task=args.task,
        filter_list=args.filter_list,
        filter_diagnosis=args.filter_diagnosis,
        include_pet_volume=args.include_pet_volume,
        num_visits=args.num_visits,
        norm=args.normalize
    )
    dataset_train = dataset_dict['train_dataset']
    dataset_test = dataset_dict['test_dataset']

    get_updates(dataset_train[0], args, dict())  # update args based on dataset slice and wandb_sweep
    args.pred_node_dim = 1 + sum(args.filter_list) if args.filter_list[0] else sum(args.filter_list)

    print("{}".format(args).replace(", ", ",\n"))

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        num_tasks = 1
        global_rank = 0
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    scaler = dataset_train.scaler
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=partial(
            collator,
            max_node=args.num_nodes,
            multi_hop_max_dist=args.multi_hop_max_dist,
            spatial_pos_max=args.num_spatial,
            graph_token=args.graph_token,
            scaler=scaler,
        ),
    )

    model = baselines.__dict__[args.model](
        **vars(args),
    )
    model = misc.load_finetune(args, model)

    if not args.finetune:
        raise ValueError('specify the finetuned model checkpoint path')
    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )
        print('From Rank: {}, ==> Making model..'.format(args.rank), force=True)
        dist.barrier()
        model_without_ddp = model.module

    criterion = None
    print("criterion = %s" % str(criterion))

    log_stats = test(data_loader_test, model, device, args, fp32=args.fp32)

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        else:
            wandb_run.log(log_stats)
        with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

    return [log_stats]
