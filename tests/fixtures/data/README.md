# Test Data Fixtures

This directory contains test data for the medical model configuration tests.

## Directory Structure

```
data/
├── datasets/               # Complete datasets for testing
│   ├── medical_qa_dataset.jsonl    # Medical Q&A dataset
│   ├── ner_dataset.jsonl           # Named entity recognition dataset
│   └── text_classification_dataset.csv  # Text classification dataset
├── inputs/                 # Example model inputs
│   ├── ner_input_1.json            # Input for NER task
│   └── text_classification_input_1.json  # Input for text classification
├── outputs/                # Expected model outputs
│   ├── ner_output_1.json           # Expected NER output
│   └── text_classification_output_1.json  # Expected classification output
└── texts/                  # Sample medical texts
    ├── clinical_note_1.txt         # Sample clinical note
    └── radiology_report_1.txt      # Sample radiology report
```

## Usage in Tests

### Loading Sample Texts

```python
def test_clinical_note_processing():
    with open("tests/fixtures/data/texts/clinical_note_1.txt", "r") as f:
        text = f.read()
    # Process the text...
```

### Using Input/Output Pairs

```python
import json

def test_ner_model():
    # Load input
    with open("tests/fixtures/data/inputs/ner_input_1.json", "r") as f:
        input_data = json.load(f)
    
    # Run model
    result = ner_model.predict(input_data["text"])
    
    # Load expected output
    with open("tests/fixtures/data/outputs/ner_output_1.json", "r") as f:
        expected_output = json.load(f)
    
    # Assert results
    assert result["entities"] == expected_output["entities"]
```

### Using Datasets

```python
import json

def test_ner_dataset():
    with open("tests/fixtures/data/datasets/ner_dataset.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            # Process each example...
```

## Adding New Test Data

1. **Text Files**: Add to `texts/` directory
2. **Structured Data**: Add to appropriate subdirectory in `datasets/`
3. **Input/Output Examples**: Add pairs to `inputs/` and `outputs/`
4. **Update this README** if adding new directories or significant files

## Data Formats

- **JSONL**: Each line is a valid JSON object
- **CSV**: Comma-separated values with header row
- **TXT**: Plain text files for unstructured data
