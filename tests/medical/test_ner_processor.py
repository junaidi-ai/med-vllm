import pytest

from medvllm.tasks import NERProcessor, NERResult


@pytest.mark.unit
def test_extract_entities_rule_based_basic():
    text = "Patient has myocardial infarction (MI). Aspirin given. Hemoglobin was low."
    proc = NERProcessor(inference_pipeline=None, config=None)

    res = proc.extract_entities(text)
    assert isinstance(res, NERResult)
    assert res.text  # non-empty
    ents = res.entities
    assert isinstance(ents, list)
    # Expect at least disease and medication spans present
    types = {e["type"] for e in ents}
    assert "disease" in types
    assert "medication" in types

    # Check a specific span roughly matches
    disease = next(e for e in ents if e["type"] == "disease")
    assert disease["text"].lower() == "myocardial infarction"
    assert 0 <= disease["start"] < disease["end"] <= len(res.text)


@pytest.mark.unit
def test_link_entities_adds_ontology_links():
    text = "Patient has myocardial infarction. Aspirin given."
    proc = NERProcessor(inference_pipeline=None, config=None)

    res = proc.extract_entities(text)
    linked = proc.link_entities(res, ontology="UMLS")
    assert isinstance(linked, NERResult)

    # Ensure for known terms, we get at least one link
    mi = next(e for e in linked.entities if e["text"].lower() == "myocardial infarction")
    assert "ontology_links" in mi
    assert isinstance(mi["ontology_links"], list)
    assert any(link.get("ontology") == "UMLS" for link in mi["ontology_links"])


@pytest.mark.unit
def test_highlight_entities_html_contains_spans():
    text = "Aspirin given after myocardial infarction."
    proc = NERProcessor(inference_pipeline=None, config=None)

    res = proc.extract_entities(text)
    html = proc.highlight_entities(res)
    # Expect at least one <span ...> tag in the output
    assert "<span" in html and "</span>" in html
    # The entity surface text should appear within the HTML
    assert "Aspirin" in html or "myocardial infarction" in html
