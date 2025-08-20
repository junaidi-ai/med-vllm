import pytest

from medvllm.tasks import NERProcessor, NERResult
from medvllm.tasks.ner_processor import lookup_in_ontology, set_fuzzy_threshold


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


@pytest.mark.unit
def test_lab_value_detection_rule_based():
    text = "Hemoglobin 13.5 g/dL was recorded."
    proc = NERProcessor(inference_pipeline=None, config=None)
    res = proc.extract_entities(text)
    ents = res.entities
    # Ensure a lab_value entity is detected and includes the measurement span
    lab_vals = [e for e in ents if e["type"] == "lab_value"]
    assert lab_vals, f"No lab_value detected in: {ents}"
    assert any("hemoglobin" in e["text"].lower() for e in lab_vals)


@pytest.mark.unit
def test_temporal_detection_dates_and_relative():
    text = "Follow-up on 2023-05-01, discharged 2 days ago."
    proc = NERProcessor(inference_pipeline=None, config=None)
    res = proc.extract_entities(text)
    ents = res.entities
    temporals = [e for e in ents if e["type"] == "temporal"]
    assert temporals, f"No temporal entities detected in: {ents}"
    # Should capture either the date or the relative phrase (ideally both)
    combined = " ".join(e["text"].lower() for e in temporals)
    assert "2023-05-01" in combined or "2 days ago" in combined


@pytest.mark.unit
def test_filtering_behavior_allow_unlisted_types():
    # A pipeline that emits one known and one unknown type
    class EmitsFooAndDisease:
        def run_inference(self, text: str, task_type: str = "ner"):
            if task_type != "ner":
                return {"entities": []}
            d_start = text.lower().find("myocardial infarction")
            foo_start = text.lower().find("aspirin")
            ents = []
            if d_start >= 0:
                ents.append(
                    {
                        "text": text[d_start : d_start + len("myocardial infarction")],
                        "type": "disease",
                        "start": d_start,
                        "end": d_start + len("myocardial infarction"),
                        "confidence": 0.9,
                    }
                )
            if foo_start >= 0:
                ents.append(
                    {
                        "text": text[foo_start : foo_start + len("Aspirin")],
                        "type": "foo",  # unlisted type
                        "start": foo_start,
                        "end": foo_start + len("Aspirin"),
                        "confidence": 0.8,
                    }
                )
            return {"entities": sorted(ents, key=lambda e: e["start"])}

    text = "Patient has myocardial infarction. Aspirin given."

    # Disallow unlisted: only 'disease' should survive
    from types import SimpleNamespace

    cfg_disallow = SimpleNamespace(medical_entity_types=["disease"], ner_allow_unlisted_types=False)
    proc = NERProcessor(inference_pipeline=EmitsFooAndDisease(), config=cfg_disallow)
    res = proc.extract_entities(text)
    types = {e["type"] for e in res.entities}
    assert "disease" in types and "foo" not in types

    # Allow unlisted: both should appear; unknown get type_id -1
    cfg_allow = SimpleNamespace(medical_entity_types=["disease"], ner_allow_unlisted_types=True)
    proc2 = NERProcessor(inference_pipeline=EmitsFooAndDisease(), config=cfg_allow)
    res2 = proc2.extract_entities(text)
    types2 = {e["type"] for e in res2.entities}
    assert "disease" in types2 and "foo" in types2
    foo_ent = next(e for e in res2.entities if e["type"] == "foo")
    assert foo_ent.get("type_id", -1) == -1


@pytest.mark.unit
def test_anatomical_structure_detection_rule_based():
    text = "There is tenderness over the heart and liver."
    proc = NERProcessor(inference_pipeline=None, config=None)
    res = proc.extract_entities(text)
    ents = res.entities
    anat = [e for e in ents if e["type"] == "anatomical_structure"]
    assert anat, f"No anatomical_structure detected in: {ents}"
    surface = " ".join(e["text"].lower() for e in anat)
    assert "heart" in surface or "liver" in surface


@pytest.mark.unit
def test_parent_type_enabling_allows_children():
    # Pipeline that emits a child type 'medication' while only parent 'treatment' is enabled
    class EmitsMedication:
        def run_inference(self, text: str, task_type: str = "ner"):
            pos = text.lower().find("aspirin")
            ents = []
            if pos >= 0:
                ents.append(
                    {
                        "text": text[pos : pos + len("Aspirin")],
                        "type": "medication",
                        "start": pos,
                        "end": pos + len("Aspirin"),
                        "confidence": 0.9,
                    }
                )
            return {"entities": ents}

    from types import SimpleNamespace

    cfg = SimpleNamespace(ner_enabled_entity_types=["treatment"], ner_allow_unlisted_types=False)
    proc = NERProcessor(inference_pipeline=EmitsMedication(), config=cfg)
    res = proc.extract_entities("Aspirin given.")
    types = {e["type"] for e in res.entities}
    assert "medication" in types


@pytest.mark.unit
def test_parent_type_enabled_regex_fallback_emits_no_children():
    from types import SimpleNamespace

    # Using regex fallback (no model pipeline) with only parent 'treatment' enabled
    cfg = SimpleNamespace(ner_enabled_entity_types=["treatment"], ner_allow_unlisted_types=False)
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    text = "Aspirin given."
    res = proc.extract_entities(text)

    # Regex fallback builds patterns only for explicitly enabled leaf types. Since only
    # the parent 'treatment' is enabled, no 'medication' (child) should be emitted.
    assert not res.entities or all(e["type"] != "medication" for e in res.entities)


@pytest.mark.unit
def test_confidence_threshold_filters_low_confidence():
    class EmitsTwoDiseases:
        def run_inference(self, text: str, task_type: str = "ner"):
            d1 = text.lower().find("hypertension")
            d2 = text.lower().find("diabetes")
            ents = []
            if d1 >= 0:
                ents.append(
                    {
                        "text": text[d1 : d1 + len("hypertension")],
                        "type": "disease",
                        "start": d1,
                        "end": d1 + len("hypertension"),
                        "confidence": 0.4,  # below threshold
                    }
                )
            if d2 >= 0:
                ents.append(
                    {
                        "text": text[d2 : d2 + len("diabetes")],
                        "type": "disease",
                        "start": d2,
                        "end": d2 + len("diabetes"),
                        "confidence": 0.9,  # above threshold
                    }
                )
            return {"entities": sorted(ents, key=lambda e: e["start"])}

    from types import SimpleNamespace

    cfg = SimpleNamespace(ner_confidence_threshold=0.5, ner_allow_unlisted_types=False)
    proc = NERProcessor(inference_pipeline=EmitsTwoDiseases(), config=cfg)
    text = "Hypertension and diabetes."
    res = proc.extract_entities(text)
    surface = " ".join(e["text"].lower() for e in res.entities)
    assert "diabetes" in surface and "hypertension" not in surface


@pytest.mark.unit
def test_lab_value_detection_new_units():
    text = "TSH 3.2 mIU/L, CRP 5 mg/L were noted."
    proc = NERProcessor(inference_pipeline=None, config=None)
    res = proc.extract_entities(text)
    lab_vals = [e for e in res.entities if e["type"] == "lab_value"]
    assert lab_vals, f"No lab_value detected in: {res.entities}"
    joined = " ".join(e["text"].lower() for e in lab_vals)
    assert "tsh" in joined and "miu/l" in joined
    assert "crp" in joined and "mg/l" in joined


@pytest.mark.unit
def test_custom_gazetteer_via_config():
    from types import SimpleNamespace

    cfg = SimpleNamespace(ner_custom_lexicon={"medication": ["apixaban"]})
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    text = "Apixaban 5 mg bid started."
    res = proc.extract_entities(text)
    meds = [e for e in res.entities if e["type"] == "medication"]
    assert meds, f"No medication detected in: {res.entities}"
    assert any(e["text"].lower() == "apixaban" for e in meds)


@pytest.mark.unit
def test_extended_gazetteer_detection_enabled():
    from types import SimpleNamespace

    # Enable extended gazetteer items like "amoxicillin" (medication) and "troponin" (test)
    cfg = SimpleNamespace(ner_enable_extended_gazetteer=True)
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    text = "Amoxicillin 500 mg started. Troponin elevated."
    res = proc.extract_entities(text)

    meds = [e for e in res.entities if e["type"] == "medication"]
    tests = [e for e in res.entities if e["type"] == "test"]
    assert any(
        e["text"].lower() == "amoxicillin" for e in meds
    ), f"Extended medication not detected: {res.entities}"
    assert any(
        "troponin" == e["text"].lower() for e in tests
    ), f"Extended test not detected: {res.entities}"


@pytest.mark.unit
def test_entity_linking_respects_config_enabled_flag():
    from types import SimpleNamespace

    cfg = SimpleNamespace(entity_linking=SimpleNamespace(enabled=False))
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    text = "Aspirin given after myocardial infarction."
    res = proc.extract_entities(text)
    linked = proc.link_entities(res, ontology="UMLS")
    # When disabled, entities should not gain 'ontology_links'
    assert all("ontology_links" not in e for e in linked.entities)


@pytest.mark.unit
def test_entity_linking_default_ontology_and_uri_field():
    from types import SimpleNamespace

    cfg = SimpleNamespace(entity_linking=SimpleNamespace(enabled=True, default_ontology="RXNORM"))
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    text = "Metformin 500 mg started."
    res = proc.extract_entities(text)
    linked = proc.link_entities(res)  # should use RXNORM by default
    # Find metformin entity and check RXNORM link with uri
    met = next(e for e in linked.entities if e["text"].lower() == "metformin")
    assert met.get("ontology_links"), f"No links attached: {met}"
    assert any(
        lk.get("ontology") == "RXNORM" and "rxnav" in lk.get("uri", "")
        for lk in met["ontology_links"]
    ), f"RXNORM link with uri missing: {met['ontology_links']}"


@pytest.mark.unit
def test_fuzzy_synonym_linking_with_custom_gazetteer():
    from types import SimpleNamespace

    # Add 'metformin hcl' to gazetteer so extractor emits that surface
    cfg = SimpleNamespace(ner_custom_lexicon={"medication": ["metformin hcl"]})
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    text = "Metformin HCl 500 mg started."
    res = proc.extract_entities(text)
    meds = [e for e in res.entities if e["type"] == "medication"]
    assert any(
        e["text"].lower() == "metformin hcl" for e in meds
    ), f"Custom surface not extracted: {meds}"

    linked = proc.link_entities(res, ontology="RXNORM")
    ent = next(e for e in linked.entities if e["text"].lower() == "metformin hcl")
    # Expect fuzzy match to RXNORM Metformin via synonyms
    assert ent.get("ontology_links"), f"No links for metformin hcl: {ent}"
    assert any(
        lk.get("ontology") == "RXNORM" and lk.get("code") == "6809" for lk in ent["ontology_links"]
    ), f"Fuzzy linking failed: {ent['ontology_links']}"


@pytest.mark.unit
def test_lookup_in_ontology_caching_behavior():
    # Clear cache by re-wrapping or calling cache_clear
    lookup_in_ontology.cache_clear()
    before = lookup_in_ontology.cache_info()
    # Two identical calls should result in one cache miss and one hit
    r1 = lookup_in_ontology("aspirin", "medication", "RXNORM")
    r2 = lookup_in_ontology("aspirin", "medication", "RXNORM")
    after = lookup_in_ontology.cache_info()
    assert after.misses == before.misses + 1
    assert after.hits == before.hits + 1
    assert r1 and r2 and r1 == r2


@pytest.mark.unit
def test_fuzzy_threshold_boundary_accept_and_reject(monkeypatch):
    from types import SimpleNamespace

    # High threshold should reject partial match (candidate vs canonical/synonym)
    cfg = SimpleNamespace(
        ner_custom_lexicon={"medication": ["metformin hydrochloride"]},
        entity_linking=SimpleNamespace(enabled=True, fuzzy_threshold=0.6),
    )
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    text = "Started metformin hydrochloride 500 mg."
    res = proc.extract_entities(text)
    linked = proc.link_entities(res, ontology="RXNORM")
    ent = next(e for e in linked.entities if e["text"].lower() == "metformin hydrochloride")
    assert not ent.get(
        "ontology_links"
    ), f"Should have rejected fuzzy link at high threshold: {ent}"

    # Lower threshold should accept same match
    set_fuzzy_threshold(0.3)
    linked2 = proc.link_entities(res, ontology="RXNORM")
    ent2 = next(e for e in linked2.entities if e["text"].lower() == "metformin hydrochloride")
    assert ent2.get("ontology_links"), f"Should have accepted fuzzy link at low threshold: {ent2}"
    # Restore default threshold for isolation
    set_fuzzy_threshold(0.5)


@pytest.mark.unit
def test_fetch_link_details_rxnorm_gated_and_mocked(monkeypatch):
    from types import SimpleNamespace
    import medvllm.tasks.ner_processor as np

    # Disabled gate: returns None
    cfg_disabled = SimpleNamespace(
        entity_linking=SimpleNamespace(external=SimpleNamespace(enabled=False))
    )
    proc_disabled = NERProcessor(inference_pipeline=None, config=cfg_disabled)
    out_disabled = proc_disabled.fetch_link_details({"ontology": "RXNORM", "code": "1191"})
    assert out_disabled is None

    # Enabled gate: mock network and expect properties
    cfg_enabled = SimpleNamespace(
        entity_linking=SimpleNamespace(external=SimpleNamespace(enabled=True, timeout=0.1))
    )
    proc_enabled = NERProcessor(inference_pipeline=None, config=cfg_enabled)

    def fake_http(url: str, timeout: float = 3.0):
        return {"properties": {"name": "Aspirin", "rxcui": "1191"}}

    monkeypatch.setattr(np, "_http_get_json", fake_http)
    out_enabled = proc_enabled.fetch_link_details({"ontology": "RXNORM", "code": "1191"})
    assert out_enabled and out_enabled.get("source") == "RXNORM"
    assert out_enabled.get("properties", {}).get("rxcui") == "1191"


@pytest.mark.unit
def test_fetch_link_details_umls_placeholder_when_key_present():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        entity_linking=SimpleNamespace(external=SimpleNamespace(enabled=True, umls_api_key="dummy"))
    )
    proc = NERProcessor(inference_pipeline=None, config=cfg)
    out = proc.fetch_link_details({"ontology": "UMLS", "code": "C0004057"})
    assert out and out.get("source") == "UMLS"
    assert "UMLS CAS flow not implemented" in out.get("note", "")
