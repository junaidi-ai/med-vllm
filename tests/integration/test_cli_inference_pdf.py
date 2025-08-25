import sys
from types import SimpleNamespace

from click.testing import CliRunner

from medvllm.cli import cli as root_cli


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class CapturingNERResult:
    def __init__(self) -> None:
        self.entities = []


class CapturingNERProcessor:
    def __init__(self, *args, **kwargs):
        self.last_text = None

    def extract_entities(self, text: str):
        self.last_text = text
        return CapturingNERResult()

    def link_entities(self, result: CapturingNERResult, ontology: str = "UMLS"):
        return result


class CapturingTextGenResult:
    def __init__(self, text: str) -> None:
        self.prompt = ""
        self.generated_text = text
        self.metadata = {}


class CapturingTextGenerator:
    def __init__(self, *args, **kwargs) -> None:
        self.last_prompt = None

    def generate(self, prompt: str, **kwargs):
        self.last_prompt = prompt
        return CapturingTextGenResult("OK")


def install_fake_pypdf(monkeypatch, page_texts):
    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage(t) for t in page_texts]

    fake_mod = SimpleNamespace(PdfReader=FakeReader)
    monkeypatch.setitem(sys.modules, "pypdf", fake_mod)


def test_pdf_auto_detection_in_ner(monkeypatch, tmp_path):
    # Arrange fake PDF file
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    # Install fake pypdf that returns two pages
    install_fake_pypdf(monkeypatch, ["PAGE1", "PAGE2"])

    # Patch NERProcessor to capture the input text
    import medvllm.cli.inference_commands as ic

    cap_proc = CapturingNERProcessor()
    monkeypatch.setattr(ic, "NERProcessor", lambda *a, **k: cap_proc)

    # Act
    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "ner",
            "--input",
            str(pdf_path),
            "--input-format",
            "auto",
            "--json-out",
        ],
    )

    # Assert
    assert res.exit_code == 0, res.output
    assert cap_proc.last_text is not None
    assert "PAGE1" in cap_proc.last_text and "PAGE2" in cap_proc.last_text
    # Ensure pages are joined with newlines by our implementation
    assert "\n\n" in cap_proc.last_text


def test_pdf_explicit_in_generate(monkeypatch, tmp_path):
    # Arrange fake PDF file
    pdf_path = tmp_path / "sample2.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE2")

    # Install fake pypdf with single page
    install_fake_pypdf(monkeypatch, ["ONLY-PAGE"])

    # Patch TextGenerator to capture prompt
    import medvllm.cli.inference_commands as ic

    cap_gen = CapturingTextGenerator()
    monkeypatch.setattr(ic, "TextGenerator", lambda *a, **k: cap_gen)

    # Act
    runner = CliRunner()
    res = runner.invoke(
        root_cli,
        [
            "inference",
            "generate",
            "--input",
            str(pdf_path),
            "--input-format",
            "pdf",
            "--model",
            "dummy-model",
        ],
    )

    # Assert
    assert res.exit_code == 0, res.output
    assert cap_gen.last_prompt is not None
    assert "ONLY-PAGE" in cap_gen.last_prompt
