# PRX Pixel Launcher

Local launcher for [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i), the pixel-space PRX text-to-image pipeline.

## Model Details

- **Resolution**: 1024px
- **Transformer**: ~7B params, torch.bfloat16
- **Text encoder**: Qwen3-VL text tower (Qwen3VLTextModel)
- **VAE**: None (pixel space)
- **Scheduler**: FlowMatchEulerDiscreteScheduler

## Features

- Pixel-space diffusion (no VAE encoding/decoding)
- Text-to-image generation
- Built-in safety filtering (NSFW + violence detection)
- Adjustable parameters: steps, guidance scale, scheduler shift, resolution, seed

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- ~14GB VRAM (for 1024px generation at bfloat16)

## Installation

1. Click **Install** in the Pinokio UI
2. Enter your Hugging Face token if required (the model may need authentication)

## Usage

1. Click **Start** to launch the Gradio web UI
2. Enter your prompt and adjust generation parameters
3. Click **Generate** to create images

## Generation Tips

- **CFG = 1.0**: Gives good, realistic results most of the time
- **Higher CFG**: Can help correct structural issues but leads to less diversity and less realistic textures
- **Scheduler shift**: Default 3.0, adjustable 1.0-8.0
- **Resolution**: 512, 768, or 1024px

## API Access

### Python
```python
import torch
from diffusers import PRXPixelPipeline

pipe = PRXPixelPipeline.from_pretrained(
    "Photoroom/prxpixel-t2i",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

image = pipe(
    "A front-facing portrait of a lion in the golden savanna at sunset.",
    num_inference_steps=28,
    guidance_scale=5.0
).images[0]
image.save("output.png")
```

## License

Apache 2.0 - See [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) for details.
