"""Adapters to connect task models to NERProcessor-compatible pipelines.

Provides MedicalNERAdapter which wraps MedicalNER and exposes a
`run_inference(text, task_type="ner") -> {"entities": [...]}` API with
character-based spans and confidence scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

try:  # Optional import; adapter is only usable if model deps are available
    from .medical_ner import MedicalNER  # type: ignore
except Exception:  # pragma: no cover - optional dependency not present
    MedicalNER = Any  # type: ignore


class MedicalNERAdapter:
    """Wraps a MedicalNER model to return character-based entity spans.

    Args:
        model: An instance of medvllm.tasks.medical_ner.MedicalNER
        label_to_type: Optional mapping from integer entity type IDs to type strings
                       (e.g., {0: "disease", 1: "medication", ...}). If not
                       provided and a `config` with `medical_entity_types` is available,
                       that list will be used by index. Otherwise, types fall back to
                       "ent_{idx}".
        config: Optional config to infer label types or thresholds, unused otherwise.
        device: Torch device string to run inference on. Defaults to CUDA if available.
    """

    def __init__(
        self,
        model: MedicalNER,
        label_to_type: Optional[Dict[int, str]] = None,
        config: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Derive label_to_type mapping if not given
        if label_to_type is not None:
            self.label_to_type = {int(k): str(v).lower() for k, v in label_to_type.items()}
        else:
            derived: Dict[int, str] = {}
            try:
                types = getattr(config, "medical_entity_types", None)
                if types:
                    for i, t in enumerate(list(types)):
                        derived[i] = str(t).lower()
            except Exception:
                pass
            self.label_to_type = derived

    def run_inference(self, text: str, task_type: str = "ner") -> Dict[str, Any]:
        if task_type != "ner":
            return {"entities": []}

        # Safe no-grad context: support when torch.no_grad is a non-callable object in tests
        _nograd_cm = None
        try:
            _nograd_cm = torch.no_grad() if callable(torch.no_grad) else torch.no_grad
        except Exception:
            _nograd_cm = torch.no_grad if hasattr(torch, "no_grad") else None

        if _nograd_cm is None:
            # Fallback: do not use a no-grad context
            return self._run_inference_impl(text)
        else:
            with _nograd_cm:
                return self._run_inference_impl(text)

    def _run_inference_impl(self, text: str) -> Dict[str, Any]:
        self.model.eval()
        self.model.to(self.device)

        # Tokenize with offsets for char-span mapping
        tok = self.model.tokenizer
        inputs = tok(
            [text],
            return_offsets_mapping=True,
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        logits = outputs["logits"][0]  # (seq_len, num_labels)
        preds = torch.argmax(logits, dim=-1)  # (seq_len,)
        offsets = inputs["offset_mapping"][0].tolist()  # List[Tuple[int,int]]

        # word_ids are available on fast tokenizers; handle if missing
        try:
            word_ids = inputs.word_ids(batch_index=0)
        except Exception:
            word_ids = [None for _ in range(len(offsets))]

        entities: List[Dict[str, Any]] = []
        current_type: Optional[int] = None
        current_start: Optional[int] = None
        current_end: Optional[int] = None
        current_score: float = 0.0
        n_tokens_in_span: int = 0

        for j, (label, off) in enumerate(zip(preds.tolist(), offsets)):
            # Skip special tokens (offset (0,0)) and non-word pieces
            if off is None or off[1] <= off[0]:
                continue
            if word_ids and word_ids[j] is None:
                continue

            # Even-odd scheme: odd labels indicate entity tokens; type = label // 2
            if label % 2 == 1:
                ent_type = label // 2
                # Probability for this label
                prob = torch.softmax(logits[j], dim=-1)[label].item()
                if current_type is not None and current_type == ent_type:
                    # continue span
                    current_end = off[1]
                    n_tokens_in_span += 1
                    current_score = (
                        current_score * (n_tokens_in_span - 1) + prob
                    ) / n_tokens_in_span
                else:
                    # flush previous
                    if (
                        current_type is not None
                        and current_start is not None
                        and current_end is not None
                    ):
                        entities.append(
                            self._make_entity(
                                text, current_type, current_start, current_end, current_score
                            )
                        )
                    # start new
                    current_type = ent_type
                    current_start = off[0]
                    current_end = off[1]
                    n_tokens_in_span = 1
                    current_score = prob
            else:
                # Non-entity token: close any open span
                if (
                    current_type is not None
                    and current_start is not None
                    and current_end is not None
                ):
                    entities.append(
                        self._make_entity(
                            text, current_type, current_start, current_end, current_score
                        )
                    )
                current_type = None
                current_start = None
                current_end = None
                n_tokens_in_span = 0
                current_score = 0.0

        # Flush at end
        if current_type is not None and current_start is not None and current_end is not None:
            entities.append(
                self._make_entity(text, current_type, current_start, current_end, current_score)
            )

        return {"entities": entities}

    def _make_entity(
        self, text: str, ent_type_idx: int, start: int, end: int, score: float
    ) -> Dict[str, Any]:
        label = self.label_to_type.get(ent_type_idx, f"ent_{ent_type_idx}")
        return {
            "text": text[start:end],
            "type": label,
            "start": int(start),
            "end": int(end),
            "confidence": float(score),
        }
