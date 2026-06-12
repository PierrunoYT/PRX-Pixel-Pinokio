#!/usr/bin/env python3
"""Local launcher for PRXPixel (pixel-space PRX) text-to-image pipeline.

PRXPixel works in RGB pixel space (no VAE). The pipeline's post-processing already handles
this, so output_type="pil" returns a ready image directly.
"""

import os
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

print(f"Loading {MODEL_ID}...")
load_kwargs = {"torch_dtype": torch.bfloat16}
if HF_TOKEN:
    load_kwargs["token"] = HF_TOKEN

pipe = PRXPixelPipeline.from_pretrained(MODEL_ID, **load_kwargs)
print(f"Loading safety classifier {SAFETY_MODEL}...")
safety_tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL)
safety_model = AutoModelForSequenceClassification.from_pretrained(SAFETY_MODEL)
_NSFW_ID = next(i for i, lbl in safety_model.config.id2label.items() if lbl.lower() == "nsfw")
print("Pipeline + safety guard ready.")


def _is_blocked(text: str) -> bool:
    """Return True if the prompt is sexual NSFW (classifier) or hits the violence blocklist."""
    low = (text or "").lower()
    if any(w in low for w in VIOLENCE_WORDS):
        return True
    inputs = safety_tokenizer(text or "", return_tensors="pt", truncation=True, max_length=512).to(safety_model.device)
    nsfw_prob = safety_model(**inputs).logits.softmax(-1)[0, _NSFW_ID].item()
    return nsfw_prob >= NSFW_THRESHOLD


@torch.inference_mode()
def generate(prompt, negative_prompt, steps, guidance_scale, shift, resolution, seed, randomize_seed):
    if randomize_seed or seed is None or int(seed) < 0:
        seed = int(torch.randint(0, MAX_SEED, (1,)).item())
    seed = int(seed)

    # Safety check first, before the (expensive) generation.
    safety_model.to("cuda" if torch.cuda.is_available() else "cpu")
    if _is_blocked(prompt):
        raise gr.Error("This prompt was blocked by the safety filter (no NSFW or violent content, please).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    
    # `shift` is a scheduler setting; rebuild the scheduler so the slider takes effect.
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=float(shift))

    res = int(resolution)
    generator = torch.Generator(device).manual_seed(seed) if torch.cuda.is_available() else torch.Generator().manual_seed(seed)
    
    image = pipe(
        prompt,
        negative_prompt=negative_prompt or "",
        height=res,
        width=res,
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
    examples_gallery = gr.Gallery(
        value=[(None, cap) for cap, _ in EXAMPLES],
        label="Examples",
        columns=4,
        height="auto",
        object_fit="cover",
        allow_preview=False,
        show_label=False,
    )

    examples_gallery.select(_load_example, None, prompt)

    run.click(
        generate,
        inputs=[prompt, negative_prompt, steps, guidance_scale, shift, resolution, seed, randomize_seed],
        outputs=[out_image, out_seed],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
