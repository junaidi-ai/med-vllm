import time
import torch
from tqdm import tqdm
from medvllm.models.adapters import BioBERTAdapter

def benchmark_model(model, batch_size=8, seq_len=128, iterations=100):
    model.eval()
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    
    # Generate test input
    input_text = ["This is a test sentence for benchmarking."] * batch_size
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            inputs = model.preprocess_biomedical_text(input_text)
            _ = model(**inputs)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(iterations), desc=f"Batch {batch_size}"):
            inputs = model.preprocess_biomedical_text(input_text)
            _ = model(**inputs)
    
    total_time = time.time() - start_time
    avg_latency = (total_time / iterations) * 1000  # ms
    tokens_per_sec = (batch_size * seq_len * iterations) / total_time
    
    return {
        "avg_latency_ms": avg_latency,
        "tokens_per_sec": tokens_per_sec,
        "batch_size": batch_size,
        "seq_len": seq_len
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}")
    
    try:
        model = BioBERTAdapter.from_pretrained("monologg/biobert_v1.1_pubmed")
        if device == "cuda":
            model = model.cuda()
        
        # Test different batch sizes
        for bs in [1, 4, 8, 16]:
            result = benchmark_model(model, batch_size=bs)
            print(f"Batch {bs}: {result['avg_latency_ms']:.2f} ms, {result['tokens_per_sec']:.0f} tokens/sec")
            
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        raise
