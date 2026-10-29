import io
import json
import random
from pathlib import Path

import streamlit as st
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Stable Diffusion Streamlit", layout="centered")

st.title("Stable Diffusion — Streamlit demo")
st.markdown(
    """
Enter a prompt and generate an image.  
The model is loaded from a local folder/file next to this script — no internet or HF token needed.  
Tip: for large models (SDXL) you generally need a GPU (CUDA).
"""
)


def discover_local_models(base_dir: Path):
    """Find diffusers model folders or single-file checkpoints in the app folder."""
    models = []
    if (base_dir / "model_index.json").is_file():
        models.append(base_dir)
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and (entry / "model_index.json").is_file():
            models.append(entry)
        elif entry.is_file() and entry.suffix.lower() in {".safetensors", ".ckpt"}:
            models.append(entry)
    return models


def resolve_model_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path.resolve()


def is_sdxl(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        data = json.loads((path / "model_index.json").read_text())
        return data.get("_class_name", "").lower().startswith("stable-diffusion-xl")
    except Exception:
        return False


# --- Sidebar / settings ---
st.sidebar.header("Model & runtime")
local_models = discover_local_models(APP_DIR)

if local_models:
    labels = [str(p.relative_to(APP_DIR)) if p != APP_DIR else "." for p in local_models]
    selected_label = st.sidebar.selectbox("Choose local model", labels)
    default_model_path = str(local_models[labels.index(selected_label)])
else:
    default_model_path = "model"
    st.sidebar.warning("No local model found next to app.py. Set the path below.")

model_path_input = st.sidebar.text_input("Model folder/file path", value=default_model_path)
device_override = st.sidebar.selectbox("Device override (auto-detect default)", ("auto", "cuda", "cpu"))

st.sidebar.markdown("**Generation settings**")
height = st.sidebar.selectbox("Height", (512, 640, 768))
width = st.sidebar.selectbox("Width", (512, 640, 768))
num_inference_steps = st.sidebar.slider("Steps", 10, 100, 28)
guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 15.0, 7.5)
seed_input = st.sidebar.number_input("Seed (0 = random)", min_value=0, step=1, value=0)
num_images = st.sidebar.slider("Images to generate", 1, 4, 1)

model_path = resolve_model_path(model_path_input)


# --- Device detection & dtype ---
def pick_device(override):
    if override == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return override


device = pick_device(device_override)
torch_dtype = torch.float16 if device == "cuda" else torch.float32

st.sidebar.write(f"Using device: **{device}** — dtype: **{torch_dtype}**")
st.sidebar.write(f"Model path: `{model_path}`")


# --- Cache the pipeline so it persists ---
@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: str, device: str, dtype: torch.dtype):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at '{path}'. Place the model folder/file next to app.py."
        )

    if path.is_file():
        if path.suffix.lower() not in {".safetensors", ".ckpt"}:
            raise ValueError(
                f"Unsupported model file '{path.name}'. Use a .safetensors/.ckpt file "
                "or a diffusers model folder."
            )
        pipe = StableDiffusionPipeline.from_single_file(str(path), torch_dtype=dtype)
    else:
        if not (path / "model_index.json").is_file():
            raise FileNotFoundError(
                f"'{path}' is not a diffusers model folder (missing model_index.json)."
            )
        pipe = DiffusionPipeline.from_pretrained(
            str(path), torch_dtype=dtype, local_files_only=True
        )

    pipe = pipe.to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


# --- UI for prompt & generate ---
prompt = st.text_area(
    "Prompt",
    value="A fantasy castle on a floating island, cinematic lighting",
    height=120,
)
generate_btn = st.button("Generate")

negative_prompt = st.text_input("Negative prompt (optional)", value="")

if generate_btn:
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        if device == "cpu" and is_sdxl(model_path):
            st.warning(
                "SDXL on CPU is very slow and may not fit in memory. "
                "Use a GPU or choose a smaller local model."
            )
        try:
            with st.spinner("Loading pipeline (cached) — may take a while on first run..."):
                pipe = load_pipeline(str(model_path), device, torch_dtype)
        except Exception as e:
            st.exception(f"Failed to load model: {e}")
            st.stop()

        if seed_input == 0:
            seed = random.randint(1, 2**31 - 1)
        else:
            seed = int(seed_input)

        generator = torch.Generator(device=device).manual_seed(seed)

        images = []
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

        model_name = model_path.name
        cols = st.columns(min(len(images), 4))
        for i, img in enumerate(images):
            col = cols[i % len(cols)]
            with col:
                st.image(
                    img,
                    use_column_width=True,
                    caption=f"Seed: {seed}  |  Model: {model_name}",
                )
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                btn_label = f"Download image {i+1}"
                st.download_button(
                    btn_label,
                    data=buf,
                    file_name=f"sd_image_{seed}_{i+1}.png",
                    mime="image/png",
                )

        st.markdown("---")
        st.write("### Generation details")
        st.write(f"- Model: `{model_path}`")
        st.write(f"- Seed: `{seed}`")
        st.write(f"- Size: {width}×{height}")
        st.write(f"- Steps: {num_inference_steps}")
        st.write(f"- Guidance scale: {guidance_scale}")

st.markdown("---")
st.write("Made with diffusers + Streamlit. Run: `streamlit run app.py`")
st.write(
    "If you hit memory/CUDA errors, try lowering `height`/`width`/`steps` or use a smaller model."
)
