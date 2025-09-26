---
title: Med vLLM Demo (Config-first)
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Med vLLM Demo (Config-first)

This Space demonstrates loading `MedicalModelConfig` from the Hub and provides a minimal UI for NER / Classification / Generation.

- Loads config from: `Junaidi-AI/med-vllm`
- Uses placeholders for inference so the UI is responsive without heavy models.
- Extend by wiring real inference pipelines or adapters later.

Run locally:

```bash
pip install -r requirements.txt
python app.py
```
