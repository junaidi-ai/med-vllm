import os
from typing import List, Dict, Sequence

import pytest

from medvllm.tasks import NERProcessor
from medvllm.utils.ner_metrics import compute_ner_metrics


def _bio_to_spans(tokens: Sequence[str], tags: Sequence[str]) -> List[Dict]:
    """Convert BIO/IOB2 tags to entity spans with character offsets over a joined text.

    - Reconstruct text by joining tokens with single spaces: text = " ".join(tokens)
    - Compute char start/end per token under that regime
    - Build spans for each contiguous B-*/I-* segment
    """
    # Precompute char offsets per token under simple whitespace-joined text
    starts = []
    pos = 0
    for i, tok in enumerate(tokens):
        starts.append(pos)
        pos += len(tok)
        if i != len(tokens) - 1:
            pos += 1  # space

    def tok_range(s_tok: int, e_tok: int) -> tuple[int, int]:
        s_char = starts[s_tok]
        last_tok_end = starts[e_tok] + len(tokens[e_tok])
        return s_char, last_tok_end

    spans = []
    cur_type = None
    cur_start = None
    for i, tag in enumerate(tags):
        if tag == 'O' or tag == 0:
            if cur_type is not None:
                s_char, e_char = tok_range(cur_start, i - 1)
                spans.append({"start": s_char, "end": e_char, "type": cur_type})
                cur_type, cur_start = None, None
            continue
        # Normalize tag string (handle int-to-name elsewhere)
        t = str(tag)
        if '-' in t:
            prefix, typ = t.split('-', 1)
        else:
            prefix, typ = 'B', t  # fallback
        typ = typ.lower()
        if prefix == 'B' or (cur_type is not None and typ != cur_type):
            if cur_type is not None:
                s_char, e_char = tok_range(cur_start, i - 1)
                spans.append({"start": s_char, "end": e_char, "type": cur_type})
            cur_type = typ
            cur_start = i
        else:
            # I-typ continuation; ensure cur_type is set
            if cur_type is None:
                cur_type = typ
                cur_start = i
    # Flush
    if cur_type is not None:
        s_char, e_char = tok_range(cur_start, len(tokens) - 1)
        spans.append({"start": s_char, "end": e_char, "type": cur_type})

    return spans


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cuda
def test_public_hf_med_ner_optional_gated():
    """
    Optional integration over public medical NER datasets via Hugging Face `datasets`.

    Gate conditions (all must be satisfied, otherwise the test is skipped):
      - ENABLE_PUBLIC_MED_NER_TESTS=1
      - HF_MED_NER_DATASET=<repo_or_script_name>
      - Either provide span-style or BIO-style inputs:
          Span mode (preferred):
            - HF_MED_NER_TEXT_COLUMN=<raw text column>
            - HF_MED_NER_ENTS_COLUMN=<entities as list of {start,end,type}>
          BIO mode (fallback):
            - HF_MED_NER_TOKENS_COLUMN=<tokens column>
            - HF_MED_NER_TAGS_COLUMN=<tags column (BIO/IOB2; str names or int ids)>
            - [optional] HF_MED_NER_TEXT_COLUMN (if not provided, text is reconstructed by joining tokens with spaces)

    Optional env vars:
      - HF_MED_NER_CONFIG=<dataset config name>
      - HF_MED_NER_SPLIT=<split name, default: test>
      - HF_MED_NER_LIMIT=<int limit of examples to evaluate, default: 200>

    Note: This test is schema-agnostic only if the dataset already provides pre-parsed
    entity spans in HF_MED_NER_ENTS_COLUMN. If your dataset uses BIO tagging, pre-convert
    to spans and expose them via a column to use this test.
    """
    if os.environ.get("ENABLE_PUBLIC_MED_NER_TESTS") != "1":
        pytest.skip("Public HF NER tests disabled; set ENABLE_PUBLIC_MED_NER_TESTS=1 to enable.")

    try:
        import datasets as hfds  # type: ignore
    except Exception:
        pytest.skip("HuggingFace datasets not installed or import failed.")

    ds_name = os.environ.get("HF_MED_NER_DATASET")
    text_col = os.environ.get("HF_MED_NER_TEXT_COLUMN")
    ents_col = os.environ.get("HF_MED_NER_ENTS_COLUMN")
    tokens_col = os.environ.get("HF_MED_NER_TOKENS_COLUMN")
    tags_col = os.environ.get("HF_MED_NER_TAGS_COLUMN")
    if not ds_name:
        pytest.skip("Missing HF_MED_NER_DATASET env var.")
    if not ((text_col and ents_col) or (tokens_col and tags_col)):
        pytest.skip("Provide span mode (TEXT+ENTS) or BIO mode (TOKENS+TAGS) env vars to run.")

    config = os.environ.get("HF_MED_NER_CONFIG")
    split = os.environ.get("HF_MED_NER_SPLIT", "test")
    limit = int(os.environ.get("HF_MED_NER_LIMIT", "200"))

    try:
        ds = hfds.load_dataset(ds_name, name=config) if config else hfds.load_dataset(ds_name)
        if split not in ds:
            pytest.skip(f"Split '{split}' not found in dataset {ds_name}.")
        subset = ds[split]
    except Exception as e:
        pytest.skip(f"Could not load dataset {ds_name}: {e}")

    # Prepare processor
    proc = NERProcessor(inference_pipeline=None, config=None)

    gold_docs: List[List[Dict]] = []
    pred_docs: List[List[Dict]] = []

    n = min(len(subset), limit)
    # Inspect features for possible int tag mapping in BIO mode
    tag_name_map = None
    if tokens_col and tags_col:
        try:
            feat = subset.features[tags_col]
            # Typical structure: Sequence(ClassLabel(names=[...]))
            if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
                tag_name_map = list(feat.feature.names)
        except Exception:
            tag_name_map = None

    for idx in range(n):
        rec = subset[idx]
        gold = []
        if ents_col and text_col:
            text = str(rec.get(text_col, ""))
            entities = rec.get(ents_col, []) or []
            try:
                gold = [
                    {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
                    for e in entities
                ]
            except Exception:
                pytest.skip(
                    f"Record {idx} does not contain span-style entities in column '{ents_col}'."
                )
        elif tokens_col and tags_col:
            tokens = list(rec.get(tokens_col, []))
            tags_raw = list(rec.get(tags_col, []))
            if not tokens or not tags_raw or len(tokens) != len(tags_raw):
                continue
            # Map int tags to names if needed
            if tag_name_map and isinstance(tags_raw[0], int):
                tags = [tag_name_map[t] for t in tags_raw]
            else:
                tags = [str(t) for t in tags_raw]
            # Build text if missing
            if text_col:
                text = str(rec.get(text_col, ""))
                if not text:
                    text = " ".join(tokens)
            else:
                text = " ".join(tokens)
            gold = _bio_to_spans(tokens, tags)
        else:
            # Should not occur due to gating
            continue

        res = proc.extract_entities(text)
        pred = [
            {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
            for e in res.entities
        ]
        gold_docs.append(gold)
        pred_docs.append(pred)

    # Compute strict and overlap metrics; just sanity-check ranges to avoid flakiness
    strict = compute_ner_metrics(gold_docs, pred_docs, match="strict")
    overlap = compute_ner_metrics(gold_docs, pred_docs, match="overlap", iou_threshold=0.5)

    for m in (strict["micro"], overlap["micro"]):
        for k in ("precision", "recall", "f1"):
            assert 0.0 <= m[k] <= 1.0
