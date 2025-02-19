import argparse


class GMAEParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--batch_size",
            default=4,
            type=int,
            help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
        )
        self.add_argument("--epochs", default=50, type=int)

        # directories
        self.add_argument(
            "--path_to_data_dir",
            default="",
            help="path to data directory",
        )
        self.add_argument(
            "--output_dir",
            default="./output_dir",
            help="path where to save, empty for no saving",
        )
        self.add_argument(
            "--log_dir",
            default="",
            help="path where to tensorboard log",
        )

        self.add_argument(
            "--device", default="cuda", help="device to use for training / testing"
        )
        self.add_argument("--seed", default=0, type=int)

        self.add_argument("--num_workers", default=10, type=int)
        self.add_argument(
            "--pin_mem",
            action="store_true",
            help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
        )
        self.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
        self.set_defaults(pin_mem=True)

        # distributed params
        self.add_argument(
            "--world_size", default=None, type=int, help="number of distributed processes"
        )
        self.add_argument("--local_rank", default=-1, type=int)
        self.add_argument("--torch_run", action="store_true")
        self.add_argument("--no_env", action="store_true")
        self.add_argument(
            "--dist_url", default=None, help="url used to set up distributed training"
        )
        self.add_argument('--dist-backend', default='nccl', type=str, help='')
        self.add_argument("--distributed", action="store_true", default=False)

        # < model init configs
        # transformer parameters
        self.add_argument("--node_feature_dim", default=None, type=int)
        self.add_argument("--num_nodes", default=512 * 9, type=int, help="numer of nodes in the graph")
        self.add_argument("--num_edges", default=512 * 3, type=int)
        self.add_argument("--num_in_degree", default=512, type=int)
        self.add_argument("--num_out_degree", default=512, type=int)
        self.add_argument("--num_spatial", default=512, type=int,
                          help="maximum relation between nodes, in this case, the shortest path distance")
        self.add_argument("--multi_hop_max_dist", default=5, type=int)
        self.add_argument("--num_edge_dis", default=128, type=int)
        self.add_argument("--static_graph", action="store_false", default=True)
        self.add_argument("--n_hist", default=12, type=int)
        self.add_argument("--decoder_embed_dim", default=504, type=int)
        self.add_argument("--decoder_depth", default=8, type=int)
        self.add_argument(
            "--use_conv",
            action="store_true",
            default=False,
            help='use convolution for time series prediction'
        )
        self.add_argument(
            "--end_channel",
            default=512,
            type=int,
            help="end prediction layer embedding dimension",
        )
        self.add_argument(
            "--graph_token",
            action="store_false",
            default=True,
            help="default is true.\n"
                 "for added tokens, no flags meaning adding graph/sep tokens. "
                 "if --graph_token is specified, "
                 "then graph token is disabled"
        )
        self.add_argument(
            "--cls_token",
            action="store_true",
            default=False,
            help="default is false,"
                 "if --cls_token is specified, "
                 "then cls token is enabled"
        )

        self.add_argument(
            "--act_fn",
            default="gelu",
            type=str,
            help="activation function",
            choices=["relu", "gelu"],
        )
        self.add_argument(
            "--sep_pos_embed",
            action="store_true",
            default=False,
        )
        self.add_argument(
            "--ablate_pos_embed",
            action="store_true",
            default=False,
            help='if specified, disable positional encoding'
        )
        self.add_argument(
            "--attention_bias",
            action="store_false",
            default=True,
            help='if specified, disable attention bias/spatial encoding'
        )
        self.add_argument(
            "--centrality_encoding",
            action="store_false",
            default=True,
            help='if specified, disable centrality encoding'
        )

        self.add_argument(
            "--dropout", type=float, default=0.1, metavar="D", help="dropout probability"
        )
        self.add_argument(
            "--trunc_init",
            action="store_true",
        )

        self.add_argument(
            "--fp32",
            action="store_true",
        )
        self.set_defaults(fp32=True)

        # dataset args
        self.add_argument(
            '--split_ratio',
            nargs=2,
            type=int,
            default=[20, 10],
            help="specify the test_valid ratios, default: (20, 10)")
        self.add_argument("--dataset_name", default="ADNI", help="name of the dataset")

        self.add_argument(
            '--task',
            default='pred',
            type=str,
            choices=['pred', 'class'],
            help='finetuning downstream task'
        )
        self.add_argument("--n_pred", default=12, type=int)
        self.add_argument(
            "--filter_list",
            nargs=3,
            default=(0, 1, 0),
            type=int,
            help='usage: --filter_list {0/1} {0/1} {0/1}'
                 'filter subjects with all MRI, AMY-BETA, TAU scans'
                 'default is filter AMY-BETA (0, 1, 0)'
        )
        self.add_argument(
            "--filter_diagnosis",
            action='store_true',
            default=False,
            help='filter only AD positive patients'
        )
        self.add_argument(
            "--pred_num_classes",
            default=3,
            type=int,
            help="number of the classification classes",
        )
        self.add_argument(
            "--include_pet_volume",
            action='store_true',
            default=False,
            help='include PET volume with SUVR in the input'
        )
        self.add_argument("--pred_per_time_step", action='store_true', default=False)
        self.add_argument(
            "--num_visits",
            default=2,
            type=int,
            help='filter number of visits for each subject'
        )
        self.add_argument("--normalize", action="store_true", help='z-score normalize the input')

        # wandb args
        self.add_argument(
            "--project_name",
            default="",
            help="Project name for wandb"
        )
        self.add_argument(
            "--wandb_run_id",
            default="",
            help="Resume wandb run"
        )
        self.add_argument(
            "--wandb_watch",
            default=False,
            action='store_true',
        )
        self.add_argument(
            "--wandb_offline",
            action="store_true",
            help="Use wandb in offline mode",
        )
