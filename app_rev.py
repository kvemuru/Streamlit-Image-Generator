import io
import random
from pathlib import Path

import streamlit as st
import torch

try:
    from diffusers import StableDiffusionPipeline
except ImportError as first_error:
    try:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
            StableDiffusionPipeline,
        )
    except ImportError as second_error:
        raise ImportError(
            f"Could not import StableDiffusionPipeline from diffusers ({first_error}). "
            "Activate the streamlit environment and run: pip install --upgrade diffusers"
        ) from second_error

APP_DIR = Path(__file__).resolve().parent
CONFIG_DIR = APP_DIR / "sd15_config"

st.set_page_config(page_title="Stable Diffusion — Local Checkpoint", layout="centered")

st.title("Stable Diffusion — Local Checkpoint")
st.markdown(
    """
Enter a prompt to generate an image using the checkpoint in this folder.  
The checkpoint is loaded locally; no Hugging Face token is required.
"""
)


def find_checkpoints(base_dir: Path):
    return sorted(
        p
        for p in base_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".ckpt", ".safetensors"}
    )


def pick_device(override: str) -> str:
    if override == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return override


# --- Sidebar / settings ---
st.sidebar.header("Model & runtime")
checkpoints = find_checkpoints(APP_DIR)

if checkpoints:
    default_name = "dreamlike-diffusion-1.0.ckpt"
    names = [p.name for p in checkpoints]
    default_index = names.index(default_name) if default_name in names else 0
    selected_name = st.sidebar.selectbox("Checkpoint", names, index=default_index)
    ckpt_path = APP_DIR / selected_name
else:
    ckpt_path = APP_DIR / "dreamlike-diffusion-1.0.ckpt"
    st.sidebar.error("No .ckpt or .safetensors file found in this folder.")

device_override = st.sidebar.selectbox(
    "Device override (auto-detect default)", ("auto", "cuda", "cpu")
)

st.sidebar.markdown("**Generation settings**")
height = st.sidebar.selectbox("Height", (256, 512, 640, 768), index=1)
width = st.sidebar.selectbox("Width", (256, 512, 640, 768), index=1)
num_inference_steps = st.sidebar.slider("Steps", 1, 100, 20)
guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 15.0, 7.5)
seed_input = st.sidebar.number_input("Seed (0 = random)", min_value=0, step=1, value=0)
num_images = st.sidebar.slider("Images to generate", 1, 4, 1)

device = pick_device(device_override)
torch_dtype = torch.float16 if device == "cuda" else torch.float32

st.sidebar.write(f"Using device: **{device}** — dtype: **{torch_dtype}**")
st.sidebar.write(f"Checkpoint: `{ckpt_path.name}`")
st.sidebar.caption(
    "Note: randomly initialized checkpoints produce noise. "
    "Use a trained Dreamlike Diffusion checkpoint for meaningful images."
)


# --- Cache the pipeline so it persists across reruns ---
@st.cache_resource(show_spinner=False)
def load_pipeline(ckpt_path: str, config_dir: str, device: str, dtype: torch.dtype):
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    load_kwargs = {"torch_dtype": dtype}
    config_path = Path(config_dir)
    if (config_path / "model_index.json").is_file():
        load_kwargs["config"] = str(config_path)

    pipe = StableDiffusionPipeline.from_single_file(str(path), **load_kwargs)
    pipe = pipe.to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
    except Exception:
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe


# --- Prompt & generate ---
prompt = st.text_area(
    "Prompt",
    value="A fantasy castle on a floating island, cinematic lighting",
    height=120,
)
negative_prompt = st.text_input("Negative prompt (optional)", value="")
generate_btn = st.button("Generate")

if generate_btn:
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    if not ckpt_path.is_file():
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()

    if device == "cpu":
        st.info("Running on CPU. Generation can take a while; lower the size/steps if needed.")

    try:
        with st.spinner("Loading pipeline — may take a while on first run..."):
            pipe = load_pipeline(str(ckpt_path), str(CONFIG_DIR), device, torch_dtype)
    except Exception as e:
        st.exception(f"Failed to load model: {e}")
        st.stop()

    seed = random.randint(1, 2**31 - 1) if seed_input == 0 else int(seed_input)
    generator = torch.Generator(device=device).manual_seed(seed)

    try:
        with st.spinner("Generating image(s)..."):
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images,
            )
            images = out.images
    except Exception as e:
        st.exception(f"Generation failed: {e}")
        st.stop()

    cols = st.columns(min(len(images), 4))
    for i, img in enumerate(images):
        with cols[i % len(cols)]:
            st.image(
                img,
                width="stretch",
                caption=f"Seed: {seed}  |  Model: {ckpt_path.name}",
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                f"Download image {i + 1}",
                data=buf,
                file_name=f"sd_image_{seed}_{i + 1}.png",
                mime="image/png",
            )

    st.markdown("---")
    st.write("### Generation details")
    st.write(f"- Checkpoint: `{ckpt_path}`")
    st.write(f"- Seed: `{seed}`")
    st.write(f"- Size: {width}×{height}")
    st.write(f"- Steps: {num_inference_steps}")
    st.write(f"- Guidance scale: {guidance_scale}")

st.markdown("---")
st.write("Run with: `streamlit run app_rev.py`")
