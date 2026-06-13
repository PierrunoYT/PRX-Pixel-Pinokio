module.exports = {
  run: [
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
          "uv pip install 'git+https://github.com/huggingface/diffusers.git@prx-pixel-pipeline'"
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
      method: "fs.write",
      params: {
        path: "ENVIRONMENT",
        content: "HF_TOKEN={{input.hf_token}}"
      }
    }
  ]
}
