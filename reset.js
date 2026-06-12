module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "rm -rf env",
          "rm -rf app/__pycache__",
          "rm -f app/.env"
        ]
      }
    }
  ]
}
