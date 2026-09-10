#!/usr/bin/env python3
"""Local launcher for PRXPixel (pixel-space PRX) text-to-image pipeline.

PRXPixel works in RGB pixel space (no VAE). The pipeline's post-processing already handles
this, so output_type="pil" returns a ready image directly.
"""

import os
import math
import threading
import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from diffusers import PRXPixelPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

MODEL_ID = os.environ.get("MODEL_ID", "Photoroom/prxpixel-t2i")
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_SEED = 2**31 - 1

# Safety classifier settings
SAFETY_MODEL = "eliasalbouzidi/distilbert-nsfw-text-classifier"
NSFW_THRESHOLD = 0.8
VIOLENCE_WORDS = [
    "gore", "gory", "decapitat", "beheading", "dismember", "mutilat", "disembowel",
    "bloodbath", "massacre", "torture", "lynching", "execution", "corpse", "bloody corpse",
    "slit throat", "stab wound", "gunshot wound", "self-harm", "suicide",
]

def _device_and_dtype():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return "cuda", dtype
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


DEVICE, DTYPE = _device_and_dtype()
pipe = None
safety_tokenizer = None
safety_model = None
_NSFW_ID = None
_scheduler_config = None
_generation_lock = threading.Lock()


def _load_pipeline():
    global pipe, _scheduler_config
    if pipe is None:
        print(f"Loading {MODEL_ID} on {DEVICE}...")
        kwargs = {"torch_dtype": DTYPE}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        loaded = PRXPixelPipeline.from_pretrained(MODEL_ID, **kwargs).to(DEVICE)
        _scheduler_config = dict(loaded.scheduler.config)
        pipe = loaded


def _is_blocked(text: str) -> bool:
    """Return True if the prompt is sexual NSFW (classifier) or hits the violence blocklist."""
    global safety_model, safety_tokenizer, _NSFW_ID
    low = (text or "").lower()
    if any(w in low for w in VIOLENCE_WORDS):
        return True
    if safety_model is None:
        print(f"Loading safety classifier {SAFETY_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(SAFETY_MODEL).to("cpu").eval()
        nsfw_id = next((int(i) for i, label in model.config.id2label.items()
                        if label.lower() == "nsfw"), None)
        if nsfw_id is None:
            raise RuntimeError("Safety classifier has no NSFW label.")
        safety_tokenizer, safety_model, _NSFW_ID = tokenizer, model, nsfw_id
    inputs = safety_tokenizer(text or "", return_tensors="pt", truncation=True, max_length=512).to(safety_model.device)
    nsfw_prob = safety_model(**inputs).logits.softmax(-1)[0, _NSFW_ID].item()
    return nsfw_prob >= NSFW_THRESHOLD


def generate(prompt, negative_prompt, steps, guidance_scale, shift, resolution, seed, randomize_seed):
    if not isinstance(prompt, str) or not prompt.strip():
        raise gr.Error("Enter a non-empty prompt.")
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise gr.Error("Negative prompt must be text.")
    steps = _number(steps, "Steps", 1, 50, integer=True)
    guidance_scale = _number(guidance_scale, "Guidance", 1, 10)
    shift = _number(shift, "Scheduler shift", 1, 8)
    resolution = _number(resolution, "Resolution", 512, 1024, integer=True)
    if resolution not in (512, 768, 1024):
        raise gr.Error("Resolution must be 512, 768, or 1024.")
    if not isinstance(randomize_seed, bool):
        raise gr.Error("Randomize seed must be true or false.")
    if randomize_seed or seed is None:
        seed = int(torch.randint(0, MAX_SEED + 1, (1,)).item())
    else:
        seed = _number(seed, "Seed", -1, MAX_SEED, integer=True)
        if seed == -1:
            seed = int(torch.randint(0, MAX_SEED + 1, (1,)).item())

    # The pipeline and scheduler mutate during inference, including API requests.
    with _generation_lock:
        return _generate(prompt, negative_prompt, steps, guidance_scale, shift, resolution, seed)


def _number(value, name, minimum, maximum, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise gr.Error(f"{name} must be a number.") from None
    if isinstance(value, bool) or not math.isfinite(number) or not minimum <= number <= maximum:
        raise gr.Error(f"{name} must be between {minimum} and {maximum}.")
    if integer and not number.is_integer():
        raise gr.Error(f"{name} must be a whole number.")
    return int(number) if integer else number


@torch.inference_mode()
def _generate(prompt, negative_prompt, steps, guidance_scale, shift, resolution, seed):

    # Safety check first, before the (expensive) generation.
    if _is_blocked(prompt):
        raise gr.Error("This prompt was blocked by the safety filter (no NSFW or violent content, please).")
    _load_pipeline()

    # `shift` is a scheduler setting; rebuild it from the pretrained config so the slider
    # takes effect without losing the model's other scheduler settings.
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_scheduler_config, shift=shift)

    res = int(resolution)
    generator = torch.Generator("cuda" if DEVICE == "cuda" else "cpu").manual_seed(seed)
    
    image = pipe(
        prompt,
        negative_prompt=negative_prompt or "",
        height=res,
        width=res,
        use_resolution_binning=False,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    ).images[0]
    return image, seed


EXAMPLES = [
    (
        "Three women — studio portrait",
        "A photograph depicts three women posing for a portrait. The composition is a medium shot, "
        "featuring the women from the chest up. The background is a textured, abstract artwork in shades "
        "of blue, purple, and orange. The lighting is soft and diffused, likely from a natural light "
        "source. The overall atmosphere is friendly and professional. The aesthetic is contemporary and "
        "slightly formal. The style is realistic, with a focus on capturing the subjects' expressions and "
        "appearances. The vibe is positive and approachable. The women are dressed in contemporary "
        "clothing. The woman in the center wears a white blazer over a blue dress, the woman on the left "
        "wears a black top with a teal necklace, and the woman on the right wears a black and white "
        "patterned top. The colors are muted and sophisticated. There are no synthetic elements visible "
        "in the image. The image is a photograph, not a graphic, poster, or collage. There is no text in "
        "the image.",
    ),
    ("Stacked cats", "A black cat sitting on top of a white cat sitting on top of an orange cat, stacked"),
    ("Papercraft fox", "A papercraft scene of a fox in a forest, layered cut paper, soft directional light"),
    ("Vintage Polaroid picnic", "A vintage Polaroid photograph of a 1970s family picnic, faded colors, light leaks"),
    ("Punk musician", "Portrait of a punk musician with green mohawk and nose ring, leather jacket, harsh flash photography"),
    ("Giant mushroom village", "A garden of giant mushrooms with tiny fairy houses built into their stems"),
    ("Japanese ceramicist", "A middle-aged Japanese ceramicist holding a freshly thrown bowl, dusty studio, side window light"),
    ("Maasai warrior", "Portrait of a Maasai warrior in traditional red shuka, intense gaze, savannah background blurred"),
]


def _load_example(evt: gr.SelectData):
    return EXAMPLES[evt.index][1]


def build_demo():
    with gr.Blocks(title="PRX Pixel") as demo:
        gr.Markdown("# PRX Pixel — pixel-space PRX text-to-image\nLocal launcher for `Photoroom/prxpixel-t2i`.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A front-facing portrait of a lion in the golden savanna at sunset.")
                negative_prompt = gr.Textbox(label="Negative prompt", value="")
                with gr.Row():
                    steps = gr.Slider(1, 50, value=28, step=1, label="Steps")
                    guidance_scale = gr.Slider(1.0, 10.0, value=1.0, step=0.5, label="Guidance (CFG)")
                gr.Markdown(
                    "ℹ️ **CFG = 1** gives good, realistic results most of the time. Raising CFG can help correct "
                    "structural issues on some prompts, but leads to much less diversity and less realistic textures."
                )
                with gr.Row():
                    shift = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="Scheduler shift")
                    resolution = gr.Dropdown([512, 768, 1024], value=1024, label="Resolution")
                with gr.Row():
                    seed = gr.Number(value=0, label="Seed", precision=0)
                    randomize_seed = gr.Checkbox(value=True, label="Randomize seed")
                run = gr.Button("Generate", variant="primary")
            with gr.Column():
                out_image = gr.Image(label="Output", type="pil")
                out_seed = gr.Number(label="Seed used", interactive=False)

        gr.Markdown("### Examples — click to load prompt")
        example_caption = gr.Textbox(visible=False)
        examples_gallery = gr.Dataset(
            components=[example_caption],
            samples=[[cap] for cap, _ in EXAMPLES],
            label="Examples",
            show_label=False,
        )

        examples_gallery.select(_load_example, None, prompt)

        run.click(
            generate,
            inputs=[prompt, negative_prompt, steps, guidance_scale, shift, resolution, seed, randomize_seed],
            outputs=[out_image, out_seed],
            api_name="generate",
            concurrency_limit=1,
        )

    return demo


if __name__ == "__main__":
    build_demo().queue().launch(server_name="127.0.0.1")
