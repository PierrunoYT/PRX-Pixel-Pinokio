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
- Prompt safety filtering (NSFW text classifier + violence keyword blocklist)
- Adjustable parameters: steps, guidance scale, scheduler shift, resolution, seed

## Requirements

- NVIDIA GPU with a driver compatible with CUDA 12.8, or a supported AMD GPU on Linux with ROCm 6.3
- Apple Silicon uses MPS. Intel Macs are not supported by the required PyTorch version.
- Other systems, including AMD GPUs on Windows, use the CPU fallback; this is much slower.
- The launcher creates a Python 3.11 environment (Diffusers `main` requires Python 3.11+).
- Substantial GPU/system memory: the ~7B transformer weights alone occupy roughly 14GB at bfloat16. The text encoder, activations, and other allocations require additional memory. This is not a 14GB total-VRAM guarantee.
- CUDA devices with bfloat16 support use bfloat16; other devices use float32, which needs more memory.

## Installation

1. Click **Install** in the Pinokio UI
2. Enter your Hugging Face token if required (the model may need authentication)

The token is stored in the ignored `ENVIRONMENT` file. Leaving the field blank preserves any existing token and settings. You can edit `HF_TOKEN` there later and restart the app.

Install and Update check dependencies and imports before enabling Start. Existing installations must run Install once to create the new completion marker. If dependency checks fail after an older installation, use Reset, then Install. Reset removes only `app/env`, preserving credentials and cached model downloads.

Update uses a fast-forward Git pull, so local divergent commits must be resolved before updating. Diffusers is installed from its upstream `main` branch because PRXPixel support is required; upstream changes can affect compatibility.

## Usage

1. Click **Start** to launch the Gradio web UI
2. Enter your prompt and adjust generation parameters
3. Click **Generate** to create images

The model and text classifier download/load on the first generation request. The UI opens before those downloads finish. Requests run one at a time. The text classifier stays on the CPU to reserve GPU memory for generation. Filtering checks the prompt, not the generated image, and can produce false positives or miss content.

## Generation Tips

- **CFG = 1.0**: Gives good, realistic results most of the time
- **Higher CFG**: Can help correct structural issues but leads to less diversity and less realistic textures
- **Scheduler shift**: Default 3.0, adjustable 1.0-8.0
- **Resolution**: 512, 768, or 1024px. Inference runs at the selected size; the default 1024px resolution matches model training. Smaller sizes can change image quality.
- **Seed**: 0 through 2147483647. Select Randomize seed, use -1, or leave the seed empty for a random seed. The returned seed records the value used; identical results across different hardware are not guaranteed.

## API Access

The running UI exposes `/generate`. Replace the example port with the URL shown by Pinokio. Parameters are prompt, negative prompt, steps (1–50), guidance (1–10), scheduler shift (1–8), resolution (512/768/1024), seed, and randomize seed. The response contains the image and seed used.

### Python client

Install `gradio_client`, then:

```python
from gradio_client import Client

image_path, seed = Client("http://127.0.0.1:7860").predict(
    "A papercraft fox in a forest", "", 28, 1.0, 3.0, 512, 123, False,
    api_name="/generate",
)
print(image_path, seed)
```

### JavaScript client

Install `@gradio/client`, then:

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");
const result = await client.predict("/generate", [
  "A papercraft fox in a forest", "", 28, 1.0, 3.0, 512, 123, false,
]);
console.log(result.data);
```

### Curl

Submit the request, then replace `EVENT_ID` with the returned `event_id` and read the result stream:

```sh
curl -X POST http://127.0.0.1:7860/gradio_api/call/generate -H "Content-Type: application/json" -d '{"data":["A papercraft fox in a forest","",28,1.0,3.0,512,123,false]}'
curl -N http://127.0.0.1:7860/gradio_api/call/generate/EVENT_ID
```

The shell quoting above is for POSIX shells. On Windows, use either client example or send the JSON body from a file with `curl.exe --data-binary @request.json`.

### Direct pipeline access

This calls Diffusers directly and does not run the launcher's prompt filter or input validation:

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

## Development checks

```sh
node --test tests/launcher.test.js
python -m unittest discover -s tests -p test_app.py -v
```

The Python regression tests mock model dependencies and need no weights or GPU. When Gradio is installed, they also build the real UI and check its API schema and configuration route. Full inference and accelerator compatibility must be checked on the target hardware.

## License

Apache 2.0 - See [Photoroom/prxpixel-t2i](https://huggingface.co/Photoroom/prxpixel-t2i) for details.
