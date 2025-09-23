import os
import gradio as gr

from medvllm.medical.config.models.medical_config import MedicalModelConfig

HF_REPO_ID = os.getenv("MEDVLLM_DEMO_REPO", "Junaidi-AI/med-vllm")


def load_config(repo_id: str):
    try:
        cfg = MedicalModelConfig.from_pretrained(repo_id)
        return cfg
    except Exception as e:
        raise gr.Error(f"Failed to load config from {repo_id}: {e}")


CFG = load_config(HF_REPO_ID)


def run_task(task_type: str, text: str):
    # This demo focuses on configuration-driven UX.
    # Inference backends can be wired later (e.g., adapters / pipelines).
    if not text.strip():
        return "", "Please enter text"

    if task_type == "ner":
        # Placeholder: echo entities based on naive regex matches for demo purposes only.
        import re

        entities = []
        for ent in CFG.medical_entity_types:
            # naive token contains match
            if re.search(rf"\b{re.escape(ent)}\b", text, flags=re.IGNORECASE):
                entities.append({"text": ent, "label": ent.upper(), "start": 0, "end": 0})
        highlighted = text
        for ent in entities:
            highlighted = highlighted.replace(ent["text"], f"[{ent['text']}:{ent['label']}]")
        return highlighted, str({"entities": entities})

    elif task_type == "classification":
        # Placeholder: pick first label if present
        labels = CFG.classification_labels or ["positive", "negative"]
        return labels[0], f"labels={labels}"

    else:  # generation
        # Placeholder: echo text with a note
        return text + "\n\n[generated: demo placeholder]", "generation demo"


def build_ui():
    with gr.Blocks(title="Med vLLM Demo (Config-first)") as demo:
        gr.Markdown(
            """
            # Med vLLM Demo (Config-first)
            This Space loads `MedicalModelConfig` from the Hub and demonstrates a simple, config-driven UI.

            - Repo: {repo}
            - Task: {task}
            - Model: {model}
            - Version: {ver}
            """.format(repo=HF_REPO_ID, task=CFG.task_type, model=CFG.model, ver=CFG.config_version)
        )
        with gr.Row():
            task = gr.Radio(
                choices=["ner", "classification", "generation"],
                value=CFG.task_type,
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
