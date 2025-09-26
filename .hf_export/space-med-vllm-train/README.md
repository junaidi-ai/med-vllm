---
title: Med vLLM Train (LoRA NER)
emoji: 🧪
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Med vLLM Train (LoRA NER)

This Space fine-tunes a token-classification model with LoRA (PEFT) using a public dataset. It pushes checkpoints to the umbrella model repo under `checkpoints/` as a Pull Request so results persist.

- Default base model: `dmis-lab/biobert-base-cased-v1.2`
- Default dataset (robust demo): `conll2003`
- Target repo for checkpoints: `Junaidi-AI/med-vllm`

You can change the base model and dataset in the UI. Medical datasets (e.g., `bc5cdr`, `ncbi_disease`) might require extra preprocessing which can be added later.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
