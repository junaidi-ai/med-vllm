import json
from typing import Any, Dict, List

from click.testing import CliRunner

from medvllm.cli import cli as root_cli


class DummyNERResult:
    def __init__(self, entities: List[Dict[str, Any]]):
        self.entities = entities


class DummyNERProcessor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.link_called = False

    def extract_entities(self, text: str) -> DummyNERResult:
        return DummyNERResult(
            [
                {"text": "HTN", "type": "CONDITION", "start": 10, "end": 13},
            ]
        )

    def link_entities(self, result: DummyNERResult, ontology: str = "UMLS") -> DummyNERResult:
        self.link_called = True
        for e in result.entities:
            e["ontology_links"] = [
                {"ontology": ontology, "code": "C0020538", "name": "Hypertension", "score": 0.91}
            ]
        return result


class DummyGenResult:
    def __init__(self, text: str) -> None:
        self.prompt = ""
        self.generated_text = text
        self.metadata = {"strategy": "beam"}


class DummyTextGenerator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def generate(self, prompt: str, **kwargs: Any) -> DummyGenResult:
        return DummyGenResult("DUMMY-OUT")


def test_cli_shows_inference_group_in_help() -> None:
    runner = CliRunner()
    result = runner.invoke(root_cli, ["--help"])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    assert "Commands:" in result.output
    # Top-level groups are listed (model, inference)
    assert "inference" in result.output


def test_inference_ner_json_and_no_link(monkeypatch: Any) -> None:
    # Monkeypatch the NERProcessor used by CLI
    import medvllm.cli.inference_commands as ic

    dummy_proc = DummyNERProcessor()
    monkeypatch.setattr(ic, "NERProcessor", lambda *a, **k: dummy_proc)

    runner = CliRunner()
    # With linking
    res1 = runner.invoke(
        root_cli, ["inference", "ner", "--text", "Patient with HTN.", "--json-out"]
    )  # type: ignore[arg-type]
    assert res1.exit_code == 0, res1.output
    data = json.loads(res1.output.strip())
    assert "entities" in data
    assert data["entities"][0]["text"] == "HTN"
    assert data["entities"][0].get("ontology_links"), "Expected linked entities"

    # Without linking
    dummy_proc2 = DummyNERProcessor()
    monkeypatch.setattr(ic, "NERProcessor", lambda *a, **k: dummy_proc2)
    res2 = runner.invoke(
        root_cli, ["inference", "ner", "--text", "Patient with HTN.", "--json-out", "--no-link"]
    )  # type: ignore[arg-type]
    assert res2.exit_code == 0, res2.output
    data2 = json.loads(res2.output.strip())
    assert data2["entities"][0]["text"] == "HTN"
    # Should not be linked
    assert not data2["entities"][0].get("ontology_links")


def test_inference_generate_uses_dummy(monkeypatch: Any) -> None:
    import medvllm.cli.inference_commands as ic

    monkeypatch.setattr(ic, "TextGenerator", DummyTextGenerator)

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "generate",
            "--text",
            "Create patient-friendly summary",
            "--model",
            "dummy-model",
            "--json-meta",
        ],
    )  # type: ignore[arg-type]
    assert res.exit_code == 0, res.output
    assert "DUMMY-OUT" in res.output
    # Metadata JSON follows; ensure it parses. Extract from last '{' to end.
    out = res.output
    start = out.rfind("{")
    assert start != -1, f"No JSON metadata found in output: {out}"
    json.loads(out[start:])


def test_inference_classification_json(monkeypatch: Any) -> None:
    # Monkeypatch transformers.pipeline to avoid network/model load
    class DummyPipeline:
        def __call__(self, text: str):
            return [{"label": "POSITIVE", "score": 0.9876}]

    def dummy_pipeline(task: str, model: str, return_all_scores: bool = False):  # type: ignore[override]
        assert task == "text-classification"
        return DummyPipeline()

    import medvllm.cli.inference_commands as ic

    # Create a dummy transformers module shim if needed
    try:
        import transformers  # type: ignore

        monkeypatch.setattr(transformers, "pipeline", dummy_pipeline, raising=True)
    except Exception:
        # If transformers isn't installed, mock the import at the module
        monkeypatch.setattr(ic, "pipeline", dummy_pipeline, raising=False)

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "classification",
            "--text",
            "The medication worked well.",
            "--json-out",
        ],
    )  # type: ignore[arg-type]
    assert res.exit_code == 0, res.output
    data = json.loads(res.output.strip())
    assert data["label"] == "POSITIVE"
    assert 0.0 <= float(data["score"]) <= 1.0
