# python import hack
import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from pathlib import Path

from main_test import get_args_parser, main

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    args.finetune = '/Users/markbai/Documents/GitHub/src/src/test_brain_class/checkpoint-00000.pth'
    args.graph_token = False
    args.cls_token = True
    args.batch_size = 1

    dataset_type = 'brain'
    if dataset_type == 'brain':
        args.dataset_type = 'brain'
        args.dataset_name = 'ADNI'
        args.filter_list = (0, 0, 1)
        args.num_visits = 1
        args.n_hist = 1
        args.n_pred = 0
        args.path_to_data_dir = '/Users/markbai/Documents/GitHub/gmae_st/gmae_st/data/brain/ADNI/subject_info_visits1_mri0_ab0_tau1_dxFalse'
        args.output_dir = './test_brain_class'
        args.pred_per_time_step = False
        args.pred_num_classes = 3
        args.task = 'class'
        args.normalize = True

    args.model = 'graph_mlp_micro'
    args.use_conv = True
    args.wandb_offline = True
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
