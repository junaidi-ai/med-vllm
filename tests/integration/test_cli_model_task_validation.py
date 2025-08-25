import json
from typing import Any

from click.testing import CliRunner

from medvllm.cli import cli as root_cli


class FakeMeta:
    def __init__(self, tasks: list[str] | None):
        self.capabilities = {"tasks": tasks or []}


class FakeRegistry:
    def __init__(self, registered: bool, tasks: list[str] | None):
        self._registered = registered
        self._meta = FakeMeta(tasks)

    def is_registered(self, name: str) -> bool:
        return self._registered

    def get_metadata(self, name: str) -> FakeMeta:
        return self._meta


class DummyTextGenResult:
    def __init__(self, text: str) -> None:
        self.prompt = ""
        self.generated_text = text
        self.metadata = {}


class DummyTextGenerator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def generate(self, prompt: str, **kwargs: Any) -> DummyTextGenResult:
        return DummyTextGenResult("GEN-OK")


def test_generate_fails_when_model_does_not_support_generation(monkeypatch):
    # Patch registry to say model is registered but does not support 'generation'
    import medvllm.cli.inference_commands as ic

    monkeypatch.setattr(ic, "get_registry", lambda: FakeRegistry(True, ["classification"]))
    monkeypatch.setattr(ic, "TextGenerator", DummyTextGenerator)

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "generate",
            "--text",
            "Hello",
            "--model",
            "dummy",
        ],
    )
    assert res.exit_code != 0
    assert "does not support task 'generation'" in res.output


def test_classification_succeeds_when_supported(monkeypatch):
    # Patch registry to say model supports 'classification'
    import medvllm.cli.inference_commands as ic

    monkeypatch.setattr(ic, "get_registry", lambda: FakeRegistry(True, ["classification"]))

    # Patch pipeline to avoid loading models
    class DummyPipeline:
        def __call__(self, text: str):
            return [{"label": "POSITIVE", "score": 0.9}]

    def dummy_pipeline(task: str, model: str, return_all_scores: bool = False):
        assert task == "text-classification"
        return DummyPipeline()

    # Either patch transformers.pipeline or ic.pipeline depending on availability
    try:
        import transformers  # type: ignore

        monkeypatch.setattr(transformers, "pipeline", dummy_pipeline, raising=True)
    except Exception:
        monkeypatch.setattr(ic, "pipeline", dummy_pipeline, raising=False)

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "classification",
            "--text",
            "ok",
            "--model",
            "dummy",
            "--json-out",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output.strip())
    assert data["label"] == "POSITIVE"


def test_generate_skips_validation_when_unregistered(monkeypatch):
    # Unregistered model => warning but proceed
    import medvllm.cli.inference_commands as ic

    monkeypatch.setattr(ic, "get_registry", lambda: FakeRegistry(False, ["generation"]))
    monkeypatch.setattr(ic, "TextGenerator", DummyTextGenerator)

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "generate",
            "--text",
            "Hello",
            "--model",
            "not-registered",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "GEN-OK" in res.output
