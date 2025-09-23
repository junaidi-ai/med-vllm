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
