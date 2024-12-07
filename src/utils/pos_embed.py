import src.utils.log as logging
import torch

logger = logging.get_logger(__name__)


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]

        old_hist_t_dim, new_hist_t_dim = checkpoint['args'].n_hist, model.hist_t_dim
        orig_tokens, new_tokens = int(pos_embed_checkpoint.shape[-2]), int(model.pos_embed.shape[-2])

        n_token_per_t = orig_tokens / old_hist_t_dim
        assert n_token_per_t == new_tokens / new_hist_t_dim, \
            f"two should be same, original n_token is {n_token_per_t}, new is {new_tokens / new_hist_t_dim}"

        num_extra_tokens = orig_tokens - new_tokens
        # class_token and dist_token are kept unchanged
        if old_hist_t_dim != new_hist_t_dim:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (old_hist_t_dim, n_token_per_t, new_hist_t_dim, n_token_per_t)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.contiguous().reshape(
                -1, old_hist_t_dim, n_token_per_t, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_hist_t_dim, n_token_per_t),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed
