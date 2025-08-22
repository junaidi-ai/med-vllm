import json
import os
from typing import Any, Dict, List, Sequence

import pytest

from medvllm.tasks import NERProcessor
from medvllm.utils.ner_metrics import compute_ner_metrics


def _bio_to_spans(tokens: Sequence[str], tags: Sequence[str]) -> List[Dict[str, Any]]:
    """Convert BIO/IOB2 tags to entity spans with character offsets over a joined text."""
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

    spans: List[Dict[str, Any]] = []
    cur_type = None
    cur_start = None
    for i, tag in enumerate(tags):
        if tag == "O" or tag == 0:
            if cur_type is not None:
                s_char, e_char = tok_range(cur_start, i - 1)
                spans.append({"start": s_char, "end": e_char, "type": cur_type})
                cur_type, cur_start = None, None
            continue
        t = str(tag)
        if "-" in t:
            prefix, typ = t.split("-", 1)
        else:
            prefix, typ = "B", t
        typ = typ.lower()
        if prefix == "B" or (cur_type is not None and typ != cur_type):
            if cur_type is not None:
                s_char, e_char = tok_range(cur_start, i - 1)
                spans.append({"start": s_char, "end": e_char, "type": cur_type})
            cur_type = typ
            cur_start = i
        else:
            if cur_type is None:
                cur_type = typ
                cur_start = i
    if cur_type is not None:
        s_char, e_char = tok_range(cur_start, len(tokens) - 1)
        spans.append({"start": s_char, "end": e_char, "type": cur_type})
    return spans


def _iter_dataset_examples(ds, split: str, limit: int):
    if split not in ds:
        return None
    subset = ds[split]
    return (subset[i] for i in range(min(len(subset), limit)))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cuda
def test_public_hf_med_ner_batch_optional_gated():
    """
    Batch optional integration across multiple public Hugging Face datasets.

    Gate: ENABLE_PUBLIC_MED_NER_TESTS=1 and `datasets` must be importable.

    Configuration via env:
      - HF_MED_NER_DATASETS_JSON: JSON array where each item is an object with keys:
          required: { "dataset": str }
          optional common: { "config": str|null, "split": str="test", "limit": int=200 }
          choose ONE mode:
            Span mode:
              { "text_col": str, "ents_col": str }
            BIO mode:
              { "tokens_col": str, "tags_col": str, "text_col": str|None }

    If HF_MED_NER_DATASETS_JSON is not provided, a reasonable default list is used:
      - jnlpba, ncbi_disease, bc5cdr, bc4chemd (BIO mode with tokens/ner_tags, split=test, limit=200)

    The test computes strict and IoU(0.5) overlap metrics and asserts values in [0,1].
    """
    if os.environ.get("ENABLE_PUBLIC_MED_NER_TESTS") != "1":
        pytest.skip("Disabled; set ENABLE_PUBLIC_MED_NER_TESTS=1 to enable.")

    try:
        import datasets as hfds  # type: ignore
    except Exception:
        pytest.skip("HuggingFace datasets not installed or import failed.")

    raw = os.environ.get("HF_MED_NER_DATASETS_JSON")
    if raw:
        try:
            cfgs = json.loads(raw)
            assert isinstance(cfgs, list)
        except Exception as e:
            pytest.skip(f"Invalid HF_MED_NER_DATASETS_JSON: {e}")
    else:
        cfgs = [
            {
                "dataset": "jnlpba",
                "split": "test",
                "limit": 200,
                "tokens_col": "tokens",
                "tags_col": "ner_tags",
            },
            {
                "dataset": "ncbi_disease",
                "split": "test",
                "limit": 200,
                "tokens_col": "tokens",
                "tags_col": "ner_tags",
            },
            {
                "dataset": "bc5cdr",
                "split": "test",
                "limit": 200,
                "tokens_col": "tokens",
                "tags_col": "ner_tags",
            },
            {
                "dataset": "bc4chemd",
                "split": "test",
                "limit": 200,
                "tokens_col": "tokens",
                "tags_col": "ner_tags",
            },
        ]

    proc = NERProcessor(inference_pipeline=None, config=None)

    for c in cfgs:
        name = c.get("dataset")
        if not name:
            # skip malformed item
            continue
        split = c.get("split", "test")
        limit = int(c.get("limit", 200))
        config = c.get("config")
        text_col = c.get("text_col")
        ents_col = c.get("ents_col")
        tokens_col = c.get("tokens_col")
        tags_col = c.get("tags_col")

        # Load dataset
        try:
            ds = hfds.load_dataset(name, name=config) if config else hfds.load_dataset(name)
        except Exception:
            # skip dataset that fails to load
            continue
        it = _iter_dataset_examples(ds, split, limit)
        if it is None:
            continue

        # Possible tag mapping for integer ClassLabel in BIO mode
        tag_name_map = None
        if tokens_col and tags_col:
            try:
                feat = ds[split].features[tags_col]
                if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
                    tag_name_map = list(feat.feature.names)
            except Exception:
                tag_name_map = None

        gold_docs: List[List[Dict[str, Any]]] = []
        pred_docs: List[List[Dict[str, Any]]] = []

        for rec in it:
            gold: List[Dict[str, Any]] = []
            # Span mode
            if text_col and ents_col:
                text = str(rec.get(text_col, ""))
                entities = rec.get(ents_col, []) or []
                try:
                    gold = [
                        {
                            "start": int(e["start"]),
                            "end": int(e["end"]),
                            "type": str(e["type"]).lower(),
                        }
                        for e in entities
                    ]
                except Exception:
                    # skip malformed record
                    continue
            # BIO mode
            elif tokens_col and tags_col:
                tokens = list(rec.get(tokens_col, []))
                tags_raw = list(rec.get(tags_col, []))
                if not tokens or not tags_raw or len(tokens) != len(tags_raw):
                    continue
                if tag_name_map and isinstance(tags_raw[0], int):
                    tags = [tag_name_map[t] for t in tags_raw]
                else:
                    tags = [str(t) for t in tags_raw]
                if text_col:
                    text = str(rec.get(text_col, ""))
                    if not text:
                        text = " ".join(tokens)
                else:
                    text = " ".join(tokens)
                gold = _bio_to_spans(tokens, tags)
            else:
                # not enough info
                continue

            res = proc.extract_entities(text)
            pred = [
                {"start": int(e["start"]), "end": int(e["end"]), "type": str(e["type"]).lower()}
                for e in res.entities
            ]
            gold_docs.append(gold)
            pred_docs.append(pred)

        if not gold_docs:
            # nothing collected; move on
            continue

        strict = compute_ner_metrics(gold_docs, pred_docs, match="strict")
        overlap = compute_ner_metrics(gold_docs, pred_docs, match="overlap", iou_threshold=0.5)
        for m in (strict["micro"], overlap["micro"]):
            for k in ("precision", "recall", "f1"):
                assert 0.0 <= m[k] <= 1.0
