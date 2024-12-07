# python import hack
import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from pathlib import Path

from baselines_test import get_args_parser, main

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    # args.finetune = '/Users/markbai/Downloads/checkpoint-00023.pth'
    # args.graph_token = False
    # args.cls_token = True
    # args.batch_size = 1
    # args.decoder_embed_dim = 64
    # args.decoder_depth = 6
    # args.model = 'graph_auto_pred_mini'
    # args.use_conv = True
    # args.wandb_offline = True
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
