import json
from typing import Any, Dict, List

from click.testing import CliRunner

from medvllm.cli import cli as root_cli


class DummyRegistry:
    def __init__(self, tasks: List[str]) -> None:
        class Meta:
            def __init__(self, tasks: List[str]) -> None:
                self.capabilities = {"tasks": tasks}

        self._meta = Meta(tasks)

    def is_registered(self, name: str) -> bool:
        return True

    def get_metadata(self, name: str) -> Any:
        return self._meta


class DummyHFTokenPipeline:
    def __call__(self, text: str):
        # Return aggregated entity results similar to HF with aggregation_strategy="simple"
        return [
            {
                "entity_group": "CONDITION",
                "score": 0.95,
                "word": "Hypertension",
                "start": 10,
                "end": 22,
            }
        ]


def test_inference_ner_model_backed_success_json(monkeypatch: Any) -> None:
    import medvllm.cli.inference_commands as ic

    # Mock registry to say model supports NER
    monkeypatch.setattr(ic, "get_registry", lambda: DummyRegistry(["ner", "classification"]))

    # Mock transformers.pipeline
    class PipelineModule:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return DummyHFTokenPipeline()

    def dummy_hf_pipeline(task: str, model: str, aggregation_strategy: str = "simple"):  # type: ignore[override]
        assert task == "token-classification"
        return DummyHFTokenPipeline()

    try:
        import transformers  # type: ignore

        monkeypatch.setattr(transformers, "pipeline", dummy_hf_pipeline, raising=True)
    except Exception:
        monkeypatch.setattr(ic, "pipeline", dummy_hf_pipeline, raising=False)

    # Replace NERProcessor with a thin wrapper that uses the provided inference_pipeline
    class TestProcessor:
        def __init__(self, inference_pipeline: Any = None, config: Any = None) -> None:
            self.pipeline = inference_pipeline

        class _Res:
            def __init__(self, entities: List[Dict[str, Any]]):
                self.entities = entities

        def extract_entities(self, text: str) -> "TestProcessor._Res":
            outs = (
                self.pipeline.run_inference(text, task_type="ner")
                if self.pipeline
                else {"entities": []}
            )
            return TestProcessor._Res(outs.get("entities", []))

        def link_entities(
            self, result: "TestProcessor._Res", ontology: str = "UMLS"
        ) -> "TestProcessor._Res":
            # No-op linking for this test
            return result

    monkeypatch.setattr(ic, "NERProcessor", TestProcessor)

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "ner",
            "--text",
            "Patient with Hypertension.",
            "--json-out",
            "--model",
            "dmis-lab/biobert-v1.1",
        ],
    )  # type: ignore[arg-type]
    assert res.exit_code == 0, res.output
    data = json.loads(res.output.strip())
    assert "entities" in data
    assert data["entities"], "Expected at least one entity"
    ent = data["entities"][0]
    assert ent["text"].lower().startswith("hyperten")
    assert ent["type"] in ("condition", "CONDITION".lower())
    assert int(ent["start"]) == 10 and int(ent["end"]) == 22


def test_inference_ner_model_invalid_task(monkeypatch: Any) -> None:
    import medvllm.cli.inference_commands as ic

    # Mock registry to say model does NOT support NER
    monkeypatch.setattr(ic, "get_registry", lambda: DummyRegistry(["classification"]))

    # Also mock transformers.pipeline to avoid import issues; should not be reached due to validation
    def dummy_hf_pipeline(task: str, model: str, aggregation_strategy: str = "simple"):  # type: ignore[override]
        return DummyHFTokenPipeline()

    try:
        import transformers  # type: ignore

        monkeypatch.setattr(transformers, "pipeline", dummy_hf_pipeline, raising=True)
    except Exception:
        monkeypatch.setattr(ic, "pipeline", dummy_hf_pipeline, raising=False)

    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "ner",
            "--text",
            "Some text",
            "--json-out",
            "--model",
            "dummy-no-ner",
        ],
    )  # type: ignore[arg-type]
    assert res.exit_code != 0
    assert "does not support task 'ner'" in res.output
