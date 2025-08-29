# Medical vLLM Benchmark Report
Generated: 2025-08-29 11:47:01

## System Information
```json
{}
```

## Benchmark Results

### biobert | -

| Batch | Seq | Avg Latency (ms) | Tokens/sec | Device | Model ID | GPU Mem Δ (MB) |
|------:|----:|-----------------:|-----------:|--------|----------|---------------:|
| N/A | N/A | 0.00 | 0.00 | - | - | - |
| N/A | N/A | 0.00 | 0.00 | - | - | - |
| N/A | N/A | 0.00 | 0.00 | - | - | - |
| N/A | N/A | 0.00 | 0.00 | - | - | - |
| N/A | N/A | 0.00 | 0.00 | - | - | - |
| 1 | 16 | 243.44 | 65.72 | cpu | - | - |
| 1 | 16 | 55.32 | 289.21 | cpu | - | - |
| 1 | 16 | 3851.30 | 4.15 | cpu | - | - |
| 1 | 16 | 194.95 | 82.07 | cpu | - | - |
| 1 | 64 | 433.48 | 147.64 | cpu | - | - |
| 1 | 64 | 85.04 | 752.60 | cpu | - | - |
| 1 | 64 | 8.13 | 7870.52 | cuda | - | 0.09521484375 |
| 1 | 128 | 361.02 | 354.55 | cpu | - | - |
| 1 | 128 | 132.47 | 966.24 | cpu | - | - |
| 1 | 128 | 425.55 | 300.79 | cpu | - | - |


### tiny_trainer | synthetic

| Batch | Seq | Avg Latency (ms) | Tokens/sec | Device | Model ID | GPU Mem Δ (MB) |
|------:|----:|-----------------:|-----------:|--------|----------|---------------:|
| 4 | 32 | 0.00 | 7261.22 | cpu | TinyModel-v1 | - |
| 8 | 64 | 0.00 | 26581.94 | cpu | TinyModel-v1 | - |


### clinicalbert_adapter | mimic_notes_sample.jsonl

| Batch | Seq | Avg Latency (ms) | Tokens/sec | Device | Model ID | GPU Mem Δ (MB) |
|------:|----:|-----------------:|-----------:|--------|----------|---------------:|
| 4 | 64 | 0.00 | 1558.66 | cpu | emilyalsentzer/Bio_ClinicalBERT | - |


### clinicalbert | -

| Batch | Seq | Avg Latency (ms) | Tokens/sec | Device | Model ID | GPU Mem Δ (MB) |
|------:|----:|-----------------:|-----------:|--------|----------|---------------:|
| 1 | 64 | 82.24 | 778.17 | cpu | - | - |
| 1 | 128 | 131.84 | 970.86 | cpu | - | - |


### unknown | textgen_small.jsonl

| Batch | Seq | Avg Latency (ms) | Tokens/sec | Device | Model ID | GPU Mem Δ (MB) |
|------:|----:|-----------------:|-----------:|--------|----------|---------------:|
| N/A | N/A | 0.00 | 0.00 | - | - | - |


### unknown | -

| Batch | Seq | Avg Latency (ms) | Tokens/sec | Device | Model ID | GPU Mem Δ (MB) |
|------:|----:|-----------------:|-----------:|--------|----------|---------------:|
| N/A | N/A | 0.00 | 0.00 | - | - | - |
| 1 | 64 | 56.18 | 1139.27 | - | - | - |
| 1 | 64 | 307.29 | 208.27 | - | - | - |
| 1 | 128 | 277.66 | 461.00 | - | - | - |
| 1 | 128 | 156.89 | 815.87 | - | - | - |
| 1 | 128 | 291.93 | 438.47 | - | - | - |
| 1 | 128 | 200.35 | 638.88 | - | - | - |
| 1 | 256 | 265.05 | 965.85 | - | - | - |
| 1 | 256 | 698.73 | 366.38 | - | - | - |
| 2 | 64 | 135.68 | 943.41 | - | - | - |
| 2 | 64 | 317.21 | 403.52 | - | - | - |
| 2 | 128 | 236.72 | 1081.45 | - | - | - |
| 2 | 128 | 724.54 | 353.33 | - | - | - |
| 2 | 256 | 614.37 | 833.37 | - | - | - |
| 2 | 256 | 931.96 | 549.38 | - | - | - |
| 4 | 64 | 701.77 | 364.79 | - | - | - |
| 4 | 64 | 255.06 | 1003.70 | - | - | - |
| 4 | 128 | 649.94 | 787.77 | - | - | - |
| 4 | 128 | 399.75 | 1280.81 | - | - | - |
| 4 | 256 | 1504.29 | 680.72 | - | - | - |
| 4 | 256 | 736.55 | 1390.27 | - | - | - |


### biobert_adapter | mimic_notes_sample.jsonl

| Batch | Seq | Avg Latency (ms) | Tokens/sec | Device | Model ID | GPU Mem Δ (MB) |
|------:|----:|-----------------:|-----------:|--------|----------|---------------:|
| 4 | 64 | 0.00 | 1326.11 | cpu | monologg/biobert_v1.1_pubmed | - |


## Aggregates (mean tokens/sec per group)

| Group | Mean tokens/sec | Count |
|-------|-----------------:|------:|
| biobert | - | 1083.35 | 10 |
| tiny_trainer | synthetic | 16921.58 | 2 |
| clinicalbert_adapter | mimic_notes_sample.jsonl | 1558.66 | 1 |
| clinicalbert | - | 874.51 | 2 |
| unknown | - | 735.33 | 20 |
| biobert_adapter | mimic_notes_sample.jsonl | 1326.11 | 1 |
