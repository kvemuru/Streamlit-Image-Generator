"""Create a randomly initialized, SD-1.5-shaped checkpoint.

Dreamlike Diffusion 1.0 is a Stable Diffusion 1.5 fine-tune, so this builds the
same UNet / VAE / CLIP-text-encoder architecture with random weights and writes
it in the original CompVis/LDM .ckpt layout.

The result is only a structural stand-in: it loads like a real checkpoint but
produces noise, not meaningful images.

Usage:
    python make_random_dreamlike_ckpt.py --output dreamlike-diffusion-1.0.ckpt
    python make_random_dreamlike_ckpt.py --output random.safetensors --safetensors --half
"""

import argparse

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel

UNET_CONFIG = dict(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    block_out_channels=(320, 640, 1280, 1280),
    layers_per_block=2,
    attention_head_dim=8,
    cross_attention_dim=768,
    transformer_layers_per_block=1,
    norm_num_groups=32,
    act_fn="silu",
)

VAE_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(128, 256, 512, 512),
    layers_per_block=2,
    act_fn="silu",
    latent_channels=4,
    norm_num_groups=32,
    sample_size=512,
    scaling_factor=0.18215,
)

TEXT_ENCODER_CONFIG = dict(
    vocab_size=49408,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=77,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    attention_dropout=0.0,
    initializer_factor=1.0,
    projection_dim=768,
)


# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
for i in range(4):
    for j in range(2):
        unet_conversion_map_layer.append(
            (f"input_blocks.{3 * i + j + 1}.0.", f"down_blocks.{i}.resnets.{j}.")
        )
        if i < 3:
            unet_conversion_map_layer.append(
                (f"input_blocks.{3 * i + j + 1}.1.", f"down_blocks.{i}.attentions.{j}.")
            )
    for j in range(3):
        unet_conversion_map_layer.append(
            (f"output_blocks.{3 * i + j}.0.", f"up_blocks.{i}.resnets.{j}.")
        )
        if i > 0:
            unet_conversion_map_layer.append(
                (f"output_blocks.{3 * i + j}.1.", f"up_blocks.{i}.attentions.{j}.")
            )
    if i < 3:
        unet_conversion_map_layer.append(
            (f"input_blocks.{3 * (i + 1)}.0.op.", f"down_blocks.{i}.downsamplers.0.conv.")
        )
        unet_conversion_map_layer.append(
            (
                f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}.",
                f"up_blocks.{i}.upsamplers.0.",
            )
        )

unet_conversion_map_layer.append(("middle_block.1.", "mid_block.attentions.0."))
for j in range(2):
    unet_conversion_map_layer.append((f"middle_block.{2 * j}.", f"mid_block.resnets.{j}."))


def convert_unet_state_dict(unet_state_dict):
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    return {v: unet_state_dict[k] for k, v in mapping.items()}


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    for j in range(2):
        vae_conversion_map.append(
            (f"encoder.down.{i}.block.{j}.", f"encoder.down_blocks.{i}.resnets.{j}.")
        )
    if i < 3:
        vae_conversion_map.append(
            (f"down.{i}.downsample.", f"down_blocks.{i}.downsamplers.0.")
        )
        vae_conversion_map.append(
            (f"up.{3 - i}.upsample.", f"up_blocks.{i}.upsamplers.0.")
        )
    for j in range(3):
        vae_conversion_map.append(
            (f"decoder.up.{3 - i}.block.{j}.", f"decoder.up_blocks.{i}.resnets.{j}.")
        )

for i in range(2):
    vae_conversion_map.append((f"mid.block_{i + 1}.", f"mid_block.resnets.{i}."))

vae_conversion_map_attn = [
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("proj_out.", "proj_attn."),
]

vae_extra_conversion_map = [
    ("to_q", "q"),
    ("to_k", "k"),
    ("to_v", "v"),
    ("to_out.0", "proj_out"),
]


def reshape_weight_for_sd(w):
    if not w.ndim == 1:
        return w.reshape(*w.shape, 1, 1)
    return w


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v

    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    keys_to_rename = {}
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                new_state_dict[k] = reshape_weight_for_sd(v)
        for weight_name, real_weight_name in vae_extra_conversion_map:
            if f"mid.attn_1.{weight_name}.weight" in k or f"mid.attn_1.{weight_name}.bias" in k:
                keys_to_rename[k] = k.replace(weight_name, real_weight_name)
    for k, v in keys_to_rename.items():
        if k in new_state_dict:
            new_state_dict[v] = reshape_weight_for_sd(new_state_dict[k])
            del new_state_dict[k]
    return new_state_dict


def build_random_components(seed: int):
    torch.manual_seed(seed)

    unet = UNet2DConditionModel(**UNET_CONFIG)
    vae = AutoencoderKL(**VAE_CONFIG)
    text_encoder = CLIPTextModel(CLIPTextConfig(**TEXT_ENCODER_CONFIG))

    unet.eval()
    vae.eval()
    text_encoder.eval()
    return unet, vae, text_encoder


def build_checkpoint(seed: int = 0, half: bool = False):
    unet, vae, text_encoder = build_random_components(seed)

    unet_state_dict = convert_unet_state_dict(unet.state_dict())
    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

    vae_state_dict = convert_vae_state_dict(vae.state_dict())
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    text_enc_dict = {}
    for k, v in text_encoder.state_dict().items():
        if not k.startswith("text_model."):
            k = "text_model." + k
        text_enc_dict["cond_stage_model.transformer." + k] = v

    state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
    if half:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    return {"state_dict": state_dict, "global_step": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dreamlike-diffusion-1.0.ckpt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--half", action="store_true", help="Save weights in fp16")
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save a .safetensors file instead of a torch .ckpt",
    )
    args = parser.parse_args()

    checkpoint = build_checkpoint(seed=args.seed, half=args.half)

    if args.safetensors or args.output.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(checkpoint["state_dict"], args.output)
    else:
        torch.save(checkpoint, args.output)

    num_tensors = len(checkpoint["state_dict"])
    num_params = sum(v.numel() for v in checkpoint["state_dict"].values())
    print(f"Wrote {args.output} ({num_tensors} tensors, {num_params:,} parameters)")


if __name__ == "__main__":
    main()
