"""Generate performance reports from benchmark results."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

def load_benchmark_results(results_dir: str) -> List[Dict]:
    """Load all benchmark result files."""
    results = []
    results_dir = Path(results_dir)
    
    for result_file in results_dir.glob("benchmark_*.json"):
        with open(result_file, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    results.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {result_file}")
    
    return results

def generate_markdown_report(results: List[Dict], output_file: str):
    """Generate a markdown report from benchmark results."""
    if not results:
        print("No benchmark results found.")
        return
    
    # Get system info from first result
    system_info = results[0].get('system_info', {})
    
    # Group results by model type
    results_by_model = {}
    for result in results:
        model_type = result.get('model_type', 'unknown')
        if model_type not in results_by_model:
            results_by_model[model_type] = []
        results_by_model[model_type].append(result)
    
    # Generate markdown
    markdown = [
        "# Medical vLLM Benchmark Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## System Information",
        "```json",
        json.dumps(system_info, indent=2),
        "```\n",
        "## Benchmark Results\n"
    ]
    
    # Add results for each model type
    for model_type, model_results in results_by_model.items():
        markdown.extend([
            f"### {model_type.upper()} Model\n",
            "| Batch Size | Sequence Length | Avg Latency (ms) | Tokens/sec | Memory (MB) |",
            "|------------|-----------------|------------------|------------|-------------|"
        ])
        
        for result in model_results:
            markdown.append(
                f"| {result.get('batch_size', 'N/A')} "
                f"| {result.get('seq_length', 'N/A')} "
                f"| {result.get('avg_latency_ms', 'N/A'):.2f} "
                f"| {result.get('tokens_per_second', 'N/A'):.2f} "
                f"| {result.get('memory_usage_mb', {}).get('gpu_mb', 'N/A')} |"
            )
        
        markdown.append("\n")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(markdown))
    
    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate benchmark reports')
    parser.add_argument('--results-dir', type=str, default='benchmark_results',
                      help='Directory containing benchmark results')
    parser.add_argument('--output', type=str, default='benchmark_report.md',
                      help='Output markdown file')
    
    args = parser.parse_args()
    
    results = load_benchmark_results(args.results_dir)
    generate_markdown_report(results, args.output)
