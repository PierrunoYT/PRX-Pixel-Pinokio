module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull --ff-only"
      }
    },
    {
      method: "fs.rm",
      params: {
        path: "app/env/.installed"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.11",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install git+https://github.com/huggingface/diffusers.git@main --upgrade-package diffusers"
        ]
      }
    },
    // Separate step so it runs even if an install above failed. Pinokio stops only on
    // output matching /Error:/, which uv does not print; RuntimeError tracebacks do.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip check || python -c \"raise RuntimeError('dependency check failed')\"",
          "python -c \"from diffusers import PRXPixelPipeline; from transformers import Qwen3VLTextModel; import gradio\""
        ]
      }
    },
    {
      method: "fs.write",
      params: {
        path: "app/env/.installed",
        text: "ok\n"
      }
    }
  ]
}
