module.exports = {
  run: [
    {
      method: "fs.rm",
      params: { path: "app/env/.installed" }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.11",
        path: "app",
        message: "python --version"
      }
    },
    // Install the backend-specific torch first; requirements (e.g. accelerate)
    // depend on torch and would otherwise pull the generic PyPI build.
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
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install git+https://github.com/huggingface/diffusers.git@main"
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
      method: "input",
      params: {
        title: "Hugging Face Token",
        description: "The PRXPixel model may require a Hugging Face token for access. Enter your HF_TOKEN (or leave blank if not needed):",
        form: [{
          type: "password",
          key: "hf_token",
          title: "HF_TOKEN",
          placeholder: "hf_..."
        }]
      }
    },
    {
      when: "{{input && input.hf_token && input.hf_token.trim()}}",
      method: "fs.append",
      params: {
        path: "ENVIRONMENT",
        text: "\nHF_TOKEN={{input.hf_token.trim().replace(/[\\r\\n]/g, '')}}\n"
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
