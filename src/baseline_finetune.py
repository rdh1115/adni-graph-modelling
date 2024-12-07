# --------------------------------------------------------
# References:
# mae_st: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

import datetime
import json
import os
import time
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
from engine_finetune import evaluate, train_one_epoch
from data.get_dataset import get_dataset
from data.utils import collator
from utils.log import master_print as print
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import get_updates
import utils.lr_decay as lrd
import utils.misc as misc
from utils.model_parser import GMAEParser
from utils.log import setup_wandb


def get_args_parser():
    parser = GMAEParser(
        "baseline fine-tuning for classification or prediction", add_help=False
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256 if batch size is >= 256 "
             "else base_lr",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-9,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        '--gcn_type',
        default='gat',
        help='type of graph convolution for classification'
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")

    # < model init configs
    # Model parameters
    parser.add_argument(
        "--model",
        default="gnn_mlp_mini",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float, default=0.1, metavar="D",
        help="classification MLP dropout probability"
    )
    # >
    parser.add_argument("--checkpoint_period", default=22, type=int)

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
        args.project_name = f'baseline_{args.task}_{args.dataset_name}_{args.model}'

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available.')
        args.device = 'cpu'
        # pytorch dataloader sometimes causes bug in multi-thread
        torch.set_num_threads(1)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    args.test = False
    task = args.task
    if task == 'class':
        args.n_pred = 0
    if args.n_hist > 1:
        args.n_hist = 1
        print('changed --n_hist to 1 for better biological interpretation')
    # start logging
    global_rank = misc.get_rank() if args.distributed else 0
    if args.log_dir:
        if not args.eval:
            if global_rank == 0:
                try:
                    pathmgr.mkdirs(args.log_dir)
                except Exception as _:
                    pass
                log_writer = SummaryWriter(log_dir=args.log_dir)
            else:
                log_writer = None
        wandb_params = None
    else:
        wandb_run, wandb_params = setup_wandb(args, global_rank, device)
        log_writer = None

    # get datasets
    dataset_dict = get_dataset(
        mode="finetune",
        data_dir=args.path_to_data_dir,
        dataset_name=args.dataset_name,
        n_hist=args.n_hist,
        n_pred=args.n_pred,
        split_ratio=tuple(args.split_ratio),
        graph_token=args.graph_token,
        seed=seed,
        task=args.task,
        filter_list=args.filter_list,
        include_pet_volume=args.include_pet_volume,
        filter_diagnosis=args.filter_diagnosis,
        num_visits=args.num_visits,
        norm=args.normalize
    )
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['valid_dataset']

    get_updates(dataset_train[0], args, wandb_params)  # update args based on dataset slice and wandb_sweep
    args.pred_node_dim = 1 + sum(args.filter_list) if args.filter_list[0] else sum(args.filter_list)

    print("{}".format(args).replace(", ", ",\n"))

    # set up dataloaders based on ddp
    if args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        num_tasks = 1
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    scaler = dataset_train.scaler
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
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
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
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
            scaler=scaler
        ),
    )

    # define the model
    model = baselines.__dict__[args.model](
        **vars(args),
        class_init_prob=dataset_dict['class_init_prob'] if task == 'class' else None,
    )
    if misc.get_last_checkpoint(args) is None and args.finetune and not args.eval:
        model = misc.load_finetune(args, model)
    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print("base lr: %.2e" % args.blr)
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            # find_unused_parameters=True,
        )
        print('From Rank: {}, ==> Making model..'.format(args.rank), force=True)
        dist.barrier()
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_baselines(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler(fp32=args.fp32)
    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    criterion = None
    if task == 'class':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
        max_accuracy = 0.0
    elif task == 'pred':
        criterion = torch.nn.MSELoss()
        min_loss = 0.0
    print("criterion = %s" % str(criterion))

    if args.eval:  # if evaluate pre-trained model first
        test_stats = evaluate(
            data_loader=data_loader_val,
            model=model,
            criterion=criterion,
            device=device,
            args=args
        )

        exit(0)

    if wandb_params is not None and args.wandb_watch and misc.is_main_process():
        wandb.watch(model, log='all')

    checkpoint_path = ""
    print(f"Start training for {args.epochs} epochs from epoch {args.start_epoch}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            dist.barrier()

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            wandb_log=wandb_run,
            args=args,
            fp32=args.fp32,
        )
        if args.output_dir and (
                epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        test_stats = evaluate(
            data_loader=data_loader_val,
            model=model,
            criterion=criterion,
            device=device,
            args=args
        )
        if task == 'class':
            print(
                f"Accuracy of the network on {len(dataset_val)} points: {test_stats['acc1']:.1f}%"
            )
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f"Max accuracy: {max_accuracy:.2f}%")
            if log_writer is not None:
                k = 5 if args.pred_num_classes > 3 else 2
                acc_k = test_stats[f"acc{k}"]
                log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
                log_writer.add_scalar(f"perf/test_acc{k}", acc_k, epoch)
                log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        elif task == 'pred':
            print(
                f"MAE of the network on {len(dataset_val)} points: {test_stats['mae']:.1f}"
            )

            if log_writer is not None:
                log_writer.add_scalar("perf/mae", test_stats["mae"], epoch)
                log_writer.add_scalar("perf/rmse", test_stats["rmse"], epoch)
                log_writer.add_scalar("perf/mape", test_stats["mape"], epoch)

        log_stats = {
            **{f"train_{k}_global_avg": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }
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

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.distributed:
        print("From Rank: {}, Training time {}".format(args.rank, total_time_str), force=True)
        dist.barrier()
    else:
        print("Training time {}".format(total_time_str))

    return [checkpoint_path]
