import os
import json
from typing import Dict, Any

import gradio as gr
from huggingface_hub import hf_hub_download

HF_REPO_ID = os.getenv("MEDVLLM_DEMO_REPO", "Junaidi-AI/med-vllm")


def _load_config_dict(repo_id: str) -> Dict[str, Any]:
    """Fetch config.json or config.yaml from a Hub repo and return as dict."""
    # Try JSON first
    try:
        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass

    # Fallback to YAML
    try:
        import yaml  # type: ignore

        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("YAML config is not a mapping/dict")
            return data
    except Exception as e:
        raise gr.Error(f"Failed to load config.json or config.yaml from {repo_id}: {e}")


CFG: Dict[str, Any] = _load_config_dict(HF_REPO_ID)


def run_task(task_type: str, text: str):
    # This demo focuses on configuration-driven UX.
    # Inference backends can be wired later (e.g., adapters / pipelines).
    if not text.strip():
        return "", "Please enter text"

    if task_type == "ner":
        # Placeholder: echo entities based on naive regex matches for demo purposes only.
        import re

        entities = []
        for ent in CFG.get("medical_entity_types", []):
            # naive token contains match
            if re.search(rf"\b{re.escape(ent)}\b", text, flags=re.IGNORECASE):
                entities.append({"text": ent, "label": ent.upper(), "start": 0, "end": 0})
        highlighted = text
        for ent in entities:
            highlighted = highlighted.replace(ent["text"], f"[{ent['text']}:{ent['label']}]")
        return highlighted, str({"entities": entities})

    elif task_type == "classification":
        # Placeholder: pick first label if present
        labels = CFG.get("classification_labels") or ["positive", "negative"]
        return labels[0], f"labels={labels}"

    else:  # generation
        # Placeholder: echo text with a note
        return text + "\n\n[generated: demo placeholder]", "generation demo"


def build_ui():
    with gr.Blocks(title="Med VLLM Demo (Config-first)") as demo:
        gr.Markdown(
            """
            # Med VLLM Demo (Config-first)
            This Space loads `MedicalModelConfig` from the Hub and demonstrates a simple, config-driven UI.

            - Repo: {repo}
            - Task: {task}
            - Model: {model}
            - Version: {ver}
            """.format(
                repo=HF_REPO_ID,
                task=CFG.get("task_type", "ner"),
                model=CFG.get("model", "unknown"),
                ver=CFG.get("config_version", "unknown"),
            )
        )
        with gr.Row():
            task = gr.Radio(
                choices=["ner", "classification", "generation"],
                value=CFG.get("task_type", "ner"),
                label="Task",
            )
            text = gr.Textbox(label="Input Text", lines=6, placeholder="Enter clinical text...")
        with gr.Row():
            primary = gr.Button("Run")
        with gr.Row():
            out_primary = gr.Textbox(label="Primary Output")
            out_secondary = gr.Textbox(label="Details")

        def _on_click(t, x):
            return run_task(t, x)

        primary.click(_on_click, inputs=[task, text], outputs=[out_primary, out_secondary])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
