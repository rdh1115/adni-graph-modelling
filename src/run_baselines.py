# python import hack
import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from pathlib import Path
from baseline_finetune import get_args_parser, main

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    # dataset_type = 'brain'
    # if dataset_type == 'brain':
    #     args.dataset_type = 'brain'
    #     args.dataset_name = 'ADNI'
    #     args.filter_list = (0, 1, 0)
    #     args.num_visits = 1
    #     args.n_hist = 1
    #     args.n_pred = 0
    #     args.output_dir = './test_brain_pretrain'
    #     args.pred_per_time_step = False
    #     args.pred_num_classes = 3
    #     args.task = 'class'
    #     args.normalize = True
    #
    # args.model = 'gnn_mlp_mini'
    # args.gcn_type = 'cheb'
    # args.wandb_offline = True
    # args.epochs = 4
    # args.num_workers = 8

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
