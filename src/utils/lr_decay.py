# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(
        model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    if there is a decoder, the number of layers is the sum of encoder and decoder layers
    start conv_layers are layer 0, end conv_layers are layer -1

    """
    param_group_names = {}
    param_groups = {}

    if hasattr(model, 'decoder_blocks'):
        num_encoder_layers = len(model.blocks.layers)
        num_layers = len(model.blocks.layers) + len(model.decoder_blocks.layers) + 2
    else:
        num_encoder_layers = len(model.blocks.layers)
        num_layers = len(model.blocks.layers) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for p_name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or p_name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id(p_name, num_layers, num_encoder_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(p_name)
        param_groups[group_name]["params"].append(p)

    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id(name, num_layers, num_encoder_layers=None):
    """
    Assign a parameter with its layer id
    """
    first_layers = [
        "mask_token",
        "cls_token_embed", "decoder_cls_token_embed", "cls_token_virtual_distance",
        "graph_token_embed", "decoder_graph_token_embed", "graph_token_virtual_distance",
        "pos_embed", "pos_embed_time",
        "pos_embed_space", "pos_embed_cls",
        "decoder_pos_embed", "decoder_pos_embed_time",
        "decoder_pos_embed_space", "decoder_pos_embed_cls"
    ]
    decoder_first_layers = [
        "decoder_embed",
    ]
    final_layers = [
        "dilated_conv", "end_conv", "fc_project", "fc_his", "fc_channel", "head"
    ]
    # TODO: dilated_conv layer decay
    if name.startswith(tuple(first_layers)):
        return 0
    elif name.startswith(tuple(decoder_first_layers)):
        return num_encoder_layers + 1
    elif name.startswith(tuple(final_layers)):
        return num_layers
    elif 'decoder' in name:
        if name.startswith("decoder_blocks"):
            if 'layer' in name:
                if 'emb_layer_norm' in name:
                    return num_encoder_layers + 1
                else:
                    return int(name.split(".")[2]) + num_encoder_layers + 2
            else:
                return num_layers
        else:
            return num_layers
    elif name.startswith("blocks"):
        if 'layer' in name:
            if 'emb_layer_norm' in name:
                return 0
            else:
                return int(name.split(".")[2]) + 1
        elif 'graph_node_feature' in name or 'graph_attn_bias' in name:
            return 0
        else:
            return num_encoder_layers
    else:
        return num_encoder_layers


def param_groups_baselines(
        model, weight_decay=0.05, no_weight_decay_list=[]
):
    """
    Parameter groups for a GNN model without layer-wise lr decay.

    Parameters:
    - model: The model instance.
    - weight_decay: Weight decay (L2 regularization) parameter.
    - no_weight_decay_list: List of parameter names that should not have weight decay applied.

    Returns:
    - List of parameter groups with specified weight decay and learning rates.
    """
    param_group_names = {}
    param_groups = {}

    for p_name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or p_name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        group_name = g_decay  # No layer-wise grouping

        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(p_name)
        param_groups[group_name]["params"].append(p)

    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())
