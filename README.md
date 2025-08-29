For benchmark quick starts (CPU/GPU adapter smokes, training smokes, report generation), see:

- `benchmarks/README.md`

# Med vLLM

Med vLLM is a project aimed at creating a specialized language model for medical applications. By leveraging the efficient [Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm) and the domain knowledge of [BioBERT](https://github.com/monologg/BioBERT) and [ClinicalBERT](https://github.com/monologg/ClinicalBERT), we provide a tool that's both powerful and resource-friendly.

## Motivation

Large language models have shown great promise in various fields, but their size and resource requirements can be prohibitive, especially in resource-constrained environments like hospitals or research labs. Med vLLM addresses this by using a lightweight inference engine while maintaining high performance on medical tasks such as analyzing clinical notes or assisting with medical research.

## Key Features

- **Efficient Inference**: Powered by [Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm) for lightweight performance.
- **Medical Expertise**: Pre-trained on medical data with [BioBERT](https://github.com/monologg/BioBERT) and [ClinicalBERT](https://github.com/monologg/ClinicalBERT).
- **Easy Integration**: Seamlessly fits into existing workflows.
- **Customizable**: Adaptable for specific medical applications.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Hugging Face Transformers library

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-github-username/med-vllm.git
   ```

2. Navigate to the project directory:
   ```bash
   cd med-vllm
   ```

3. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start Example

Run a sample inference:

```bash
python run_inference.py --model bioBERT --input "The patient has a history of diabetes and hypertension."
```

This will process the input using the specified model (e.g., BioBERT). You can also use `--model clinicalBERT` to switch models.

## Testing

Med vLLM includes a comprehensive test suite to ensure code quality and functionality. The test suite is built using Python's `unittest` framework.

### Running Tests

To run all tests:

```bash
# Run all tests
python -m pytest tests/unit/ -v

# Run a specific test file
python -m pytest tests/unit/test_medical_adapters.py -v

# Run a specific test class
python -m pytest tests/unit/test_medical_adapters.py::TestBioBERTAdapter -v

# Run a specific test method
python -m pytest tests/unit/test_medical_adapters.py::TestBioBERTAdapter::test_biomedical_text_processing -v
```

### Test Coverage

To generate a test coverage report:

```bash
# Install coverage if not already installed
pip install coverage

# Run tests with coverage
coverage run -m pytest tests/unit/

# Generate coverage report
coverage report -m

# Generate HTML coverage report
coverage html
```

The HTML report will be available in the `htmlcov` directory.

### Quality & Evaluation

- A/B smoke for text generation strategies (offline echo engine):

  ```bash
  python scripts/ab_test_textgen.py --dataset benchmarks/datasets/textgen_small.jsonl --output benchmarks/results/textgen_ab_results.json
  ```

- Domain expert evaluation protocol and template: see `docs/expert_eval_protocol.md` and `docs/expert_eval_template.csv`.
  Aggregate filled scores with:

  ```bash
  python scripts/aggregate_expert_eval.py path/to/your_eval.csv
  ```

### Test Structure

The test suite is organized as follows:

- `tests/unit/test_medical_adapters.py`: Contains all unit tests for medical adapters
  - `TestBaseAdapter`: Tests for the base adapter functionality
  - `TestBioBERTAdapter`: Tests specific to BioBERT adapter
  - `TestClinicalBERTAdapter`: Tests specific to ClinicalBERT adapter

## Usage

### Text Classification

Classify a clinical note as positive or negative for a condition:

```bash
python run_inference.py --model clinicalBERT --task classify --input "Patient shows signs of pneumonia."
```

### Named Entity Recognition

Extract medical entities from text:

```bash
python run_inference.py --model bioBERT --task ner --input "Patient prescribed metformin for diabetes."
```

### NERProcessor (Lightweight utility)

Use a simple, pluggable NER processor with a regex fallback or your own model-backed pipeline:

```python
from medvllm.tasks import NERProcessor

proc = NERProcessor(inference_pipeline=None, config=None)  # regex fallback
res = proc.extract_entities("Patient has myocardial infarction (MI). Aspirin given.")
linked = proc.link_entities(res, ontology="UMLS")
html = proc.highlight_entities(linked)
```

- Example script: `examples/ner_processor_example.py`
- Documentation: `docs/ner_processor.md`

#### Benchmarking Ontology Linking

Measure linking performance and cache effectiveness on longer notes:

```bash
python3 -m benchmarks.benchmark_linking --paragraphs 50 --runs 3 --ontology RXNORM
```

See `docs/ner_processor.md` for external enrichment (RxNorm, UMLS CAS/TGT) configuration.

### Text Generation

Generate a summary of a patient's medical history:

```bash
python run_inference.py --model clinicalBERT --task generate --input "Patient has diabetes and hypertension."
```

### Fine-Tuning

To fine-tune Med vLLM on your own medical dataset:

1. Prepare your dataset in a compatible format (e.g., JSON or CSV).
2. Use the provided training script:
   ```bash
   python train.py --model bioBERT --dataset path/to/your/data
   ```
3. Evaluate the fine-tuned model with:
   ```bash
   python evaluate.py --model path/to/finetuned/model
   ```

Detailed instructions will be provided as the project evolves.

## Limitations

- Currently supports only English-language medical texts.
- Multilingual support is planned for future releases.

## Contributing

We welcome contributions! To get involved:

- Report bugs or suggest features by opening an issue.
- Submit pull requests with improvements, following the project's code style and including tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Med vLLM builds upon:

- [Nano vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- [BioBERT](https://github.com/monologg/BioBERT)
- [ClinicalBERT](https://github.com/monologg/ClinicalBERT)

Thanks to their creators for their open-source contributions.

## Citation

If you use Med vLLM in your research or application, please cite it as:

```
[SHA888](https://github.com/SHA888). (2025). Med vLLM: A Medical Language Model. GitHub repository, https://github.com/SHA888/med-vllm
```