# python import hack
import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from pathlib import Path
from main_pretrain import get_args_parser, main

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    # args.dataset_type = 'brain'
    # args.dataset_name = 'ADNI'
    #
    # args.model = 'mae_graph_mini'
    # args.decoder_embed_dim = 64
    # args.decoder_depth = 4
    # args.graph_token = False
    # args.filter_list = (1, 0, 0)
    # args.cls_token = True
    # args.normalize = True
    # args.n_hist = 2
    # args.n_pred = 0
    # args.output_dir = './test_brain_pretrain'
    # args.epochs = 2
    # args.num_workers = 8

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
