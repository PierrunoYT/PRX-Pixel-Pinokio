"""Regression tests without model downloads or a GPU."""
import importlib.util
from pathlib import Path
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def load_app(real_gradio=None):
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.backends.mps.is_available.return_value = False
    torch.inference_mode.return_value = lambda function: function
    gradio = real_gradio or MagicMock()
    if real_gradio is None:
        gradio.Error = ValueError
    dependencies = {name: MagicMock() for name in (
        "transformers", "diffusers", "diffusers.schedulers")}
    dependencies.update(torch=torch, gradio=gradio)
    spec = importlib.util.spec_from_file_location(
        "prx_app", Path(__file__).resolve().parents[1] / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(app)
    return app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app()
        self.args = ["A cat", "", 28, 1.0, 3.0, 512, 123, False]

    def test_import_does_not_load_models(self):
        self.app.PRXPixelPipeline.from_pretrained.assert_not_called()
        self.app.AutoModelForSequenceClassification.from_pretrained.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is not installed")
    def test_real_gradio_ui_and_api_schema(self):
        import gradio
        from fastapi.testclient import TestClient
        app = load_app(real_gradio=gradio)
        with app.build_demo() as demo:
            demo.queue()
            api = demo.get_api_info()["named_endpoints"]["/generate"]
            self.assertEqual(len(api["parameters"]), 8)
            self.assertEqual(len(api["returns"]), 2)
            client = TestClient(gradio.routes.App.create_app(demo))
            self.assertEqual(client.get("/config").status_code, 200)
            self.assertTrue(any(item["api_name"] == "generate"
                                for item in demo.config["dependencies"]))
        app.PRXPixelPipeline.from_pretrained.assert_not_called()

    def test_device_and_dtype_selection(self):
        torch = self.app.torch
        self.assertEqual(self.app._device_and_dtype(), ("cpu", torch.float32))
        torch.backends.mps.is_available.return_value = True
        self.assertEqual(self.app._device_and_dtype(), ("mps", torch.float32))
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = False
        self.assertEqual(self.app._device_and_dtype(), ("cuda", torch.float32))
        torch.cuda.is_bf16_supported.return_value = True
        self.assertEqual(self.app._device_and_dtype(), ("cuda", torch.bfloat16))

    def test_invalid_requests_fail_before_model_loading(self):
        for index, value in [(0, " "), (0, None), (1, []), (2, 0), (2, 51),
                             (2, 2.5), (2, True), (3, float("nan")),
                             (4, float("inf")), (5, 600), (6, 2**32),
                             (6, "no"), (6, 0.5), (7, "false")]:
            with self.subTest(index=index, value=value):
                args = self.args.copy()
                args[index] = value
                with self.assertRaises(ValueError):
                    self.app.generate(*args)
        self.app.PRXPixelPipeline.from_pretrained.assert_not_called()

    def test_seed_randomization_and_fixed_seed(self):
        self.app._generate = MagicMock(return_value=("image", 123))
        self.app.torch.randint.return_value.item.return_value = 42
        for seed, randomize, expected in [(123, False, 123), (-1, False, 42),
                                         (None, False, 42), (123, True, 42)]:
            args = self.args.copy()
            args[6:8] = [seed, randomize]
            self.app.generate(*args)
            self.assertEqual(self.app._generate.call_args.args[-1], expected)

    def test_blocked_prompt_does_not_load_pipeline(self):
        self.args[0] = "gory scene"
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.app.generate(*self.args)
        self.app.PRXPixelPipeline.from_pretrained.assert_not_called()

    def test_violence_blocklist_matches_word_starts_only(self):
        self.app.safety_model = MagicMock()
        self.app.safety_model.return_value.logits.softmax.return_value.__getitem__.return_value.item.return_value = 0.0
        self.app.safety_tokenizer = MagicMock()
        self.app._NSFW_ID = 1
        for text in ["A category of birds", "An allegory painting", "Portrait of Gregory"]:
            self.assertFalse(self.app._is_blocked(text), text)
        for text in ["Gory scene", "a mutilated statue", "SELF-HARM"]:
            self.assertTrue(self.app._is_blocked(text), text)

    def test_resolution_scheduler_config_and_seed_reach_pipeline(self):
        app = self.app
        app._is_blocked = MagicMock(return_value=False)
        app._load_pipeline = MagicMock()
        app.pipe = MagicMock(return_value=SimpleNamespace(images=["image"]))
        app._scheduler_config = {"shift": 3.0, "num_train_timesteps": 1000}
        for resolution in [512, 768, 1024]:
            self.args[5] = resolution
            self.assertEqual(app.generate(*self.args), ("image", 123))
            kwargs = app.pipe.call_args.kwargs
            self.assertEqual((kwargs["height"], kwargs["width"]), (resolution, resolution))
            self.assertFalse(kwargs["use_resolution_binning"])
        app.FlowMatchEulerDiscreteScheduler.from_config.assert_called_with(
            app._scheduler_config, shift=3.0)
        app.torch.Generator.return_value.manual_seed.assert_called_with(123)
        self.assertEqual(app._scheduler_config["num_train_timesteps"], 1000)

    def test_pipeline_is_loaded_once_and_failed_load_can_retry(self):
        app = self.app
        loaded = MagicMock()
        loaded.scheduler.config = {"shift": 3.0}
        app.PRXPixelPipeline.from_pretrained.return_value.to.side_effect = [RuntimeError("OOM"), loaded]
        with self.assertRaises(RuntimeError):
            app._load_pipeline()
        self.assertIsNone(app.pipe)
        app._load_pipeline()
        app._load_pipeline()
        self.assertEqual(app.PRXPixelPipeline.from_pretrained.call_count, 2)
        self.assertEqual(app._scheduler_config, {"shift": 3.0})

    def test_classifier_runs_on_cpu_and_accepts_string_label_ids(self):
        app = self.app
        model = MagicMock()
        model.config.id2label = {"0": "safe", "1": "NSFW"}
        model.return_value.logits.softmax.return_value.__getitem__.return_value.item.return_value = 0.9
        app.AutoModelForSequenceClassification.from_pretrained.return_value.to.return_value.eval.return_value = model
        self.assertTrue(app._is_blocked("test prompt"))
        self.assertEqual(app._NSFW_ID, 1)
        app.AutoModelForSequenceClassification.from_pretrained.return_value.to.assert_called_with("cpu")

    def test_concurrent_requests_do_not_overlap_inference(self):
        active = 0
        peak = 0
        counter_lock = threading.Lock()

        def infer(*args):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with counter_lock:
                active -= 1
            return "image", args[-1]

        self.app._generate = infer
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: self.app.generate(*self.args), range(3)))
        self.assertEqual(peak, 1)
        self.assertEqual(results, [("image", 123)] * 3)


if __name__ == "__main__":
    unittest.main()
