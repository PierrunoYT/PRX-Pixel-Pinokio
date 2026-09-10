// Install matching wheels and their dependencies; no unused attention extensions.
module.exports = {
  run: [
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: "python -c \"raise SystemExit('PRX Pixel requires Apple Silicon on macOS; current PyTorch wheels do not support Intel Macs.')\""
      },
      next: null
    },
    {
      when: "{{gpu === 'nvidia' && (platform === 'win32' || platform === 'linux')}}",
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: "uv pip install torch==2.7.0 torchvision==0.22.0 --reinstall-package torch --reinstall-package torchvision --index-url https://download.pytorch.org/whl/cu128"
      },
      next: null
    },
    {
      when: "{{gpu === 'amd' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: "uv pip install torch==2.7.0 torchvision==0.22.0 --reinstall-package torch --reinstall-package torchvision --index-url https://download.pytorch.org/whl/rocm6.3"
      },
      next: null
    },
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: "uv pip install torch==2.7.0 torchvision==0.22.0 --reinstall-package torch --reinstall-package torchvision"
      },
      next: null
    },
    {
      method: "shell.run",
      params: {
        venv: "{{args.venv}}",
        path: "{{args.path}}",
        message: "uv pip install torch==2.7.0 torchvision==0.22.0 --reinstall-package torch --reinstall-package torchvision --index-url https://download.pytorch.org/whl/cpu"
      }
    }
  ]
}
