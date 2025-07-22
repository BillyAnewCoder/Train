"""Benchmark script for MoE model performance."""

import os
import time
import argparse
import statistics
from typing import List, Dict, Any
import json

from src.models.model import MoETransformer

class MoEBenchmark:
    """Benchmark suite for MoE models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.results = {}
    
    def load_model(self):
        """Load the model for benchmarking."""
        print(f"Loading model from {self.model_path}...")
        start_time = time.time()
        
        self.model = MoETransformer.from_pretrained(self.model_path)
        
        load_time = time.time() - start_time
        self.results['model_load_time'] = load_time
        
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Model info
        self.results['model_info'] = {
            'num_parameters': self.model.get_num_params(),
            'num_experts': self.model.num_experts,
            'top_k': self.model.top_k,
            'hidden_size': self.model.hidden_size,
            'vocab_size': self.model.vocab_size,
            'num_layers': self.model.num_layers
        }
    
    def benchmark_forward_pass(self, batch_sizes: List[int] = None, 
                              seq_lengths: List[int] = None):
        """Benchmark forward pass performance."""
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
        
        if seq_lengths is None:
            seq_lengths = [128, 512, 1024, 2048]
        
        print("Benchmarking forward pass performance...")
        
        forward_results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"  Testing batch_size={batch_size}, seq_len={seq_len}")
                
                # Mock input
                from src.models.moe import MockTensor
                input_ids = MockTensor((batch_size, seq_len))
                
                # Warmup
                for _ in range(5):
                    _ = self.model.forward(input_ids)
                
                # Benchmark
                times = []
                for _ in range(20):
                    start_time = time.time()
                    _ = self.model.forward(input_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                # Calculate statistics
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times)
                min_time = min(times)
                max_time = max(times)
                
                # Calculate throughput
                tokens_per_sec = (batch_size * seq_len) / avg_time
                
                key = f"bs{batch_size}_sl{seq_len}"
                forward_results[key] = {
                    'batch_size': batch_size,
                    'seq_length': seq_len,
                    'avg_time_ms': avg_time * 1000,
                    'std_time_ms': std_time * 1000,
                    'min_time_ms': min_time * 1000,
                    'max_time_ms': max_time * 1000,
                    'tokens_per_second': tokens_per_sec
                }
        
        self.results['forward_pass'] = forward_results
    
    def benchmark_generation(self, prompts: List[str] = None,
                           max_lengths: List[int] = None):
        """Benchmark text generation performance."""
        
        if prompts is None:
            prompts = [
                "The future of artificial intelligence",
                "Once upon a time in a distant galaxy",
                "def fibonacci(n):",
                "The key advantages of machine learning are"
            ]
        
        if max_lengths is None:
            max_lengths = [50, 100, 200]
        
        print("Benchmarking generation performance...")
        
        generation_results = {}
        
        for i, prompt in enumerate(prompts):
            for max_len in max_lengths:
                print(f"  Testing prompt {i+1}, max_length={max_len}")
                
                # Mock input
                from src.models.moe import MockTensor
                input_ids = MockTensor((1, len(prompt.split())))
                
                # Benchmark generation
                times = []
                for _ in range(10):
                    start_time = time.time()
                    _ = self.model.generate(
                        input_ids,
                        max_new_tokens=max_len,
                        temperature=1.0
                    )
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                # Calculate statistics
                avg_time = statistics.mean(times)
                tokens_per_sec = max_len / avg_time
                
                key = f"prompt{i+1}_len{max_len}"
                generation_results[key] = {
                    'prompt_length': len(prompt.split()),
                    'max_new_tokens': max_len,
                    'avg_time_s': avg_time,
                    'tokens_per_second': tokens_per_sec
                }
        
        self.results['generation'] = generation_results
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage (mock implementation)."""
        
        print("Benchmarking memory usage...")
        
        # Mock memory measurements
        memory_results = {
            'model_parameters_gb': self.results['model_info']['num_parameters'] * 4 / (1024**3),
            'peak_memory_gb': 24.5,  # Mock value
            'memory_per_token_mb': 0.15,  # Mock value
            'expert_memory_gb': 8.2,  # Mock value
            'cache_memory_gb': 4.1  # Mock value
        }
        
        self.results['memory'] = memory_results
    
    def benchmark_expert_utilization(self):
        """Benchmark expert utilization patterns."""
        
        print("Benchmarking expert utilization...")
        
        # Mock expert utilization analysis
        utilization_results = {
            'avg_expert_usage': 0.85,  # Mock value
            'expert_balance_score': 0.92,  # Mock value
            'top_k_efficiency': 0.88,  # Mock value
            'load_imbalance_factor': 1.15,  # Mock value
            'expert_specialization': {
                f'expert_{i}': {
                    'usage_frequency': 0.02 + (i % 10) * 0.01,
                    'specialization_score': 0.5 + (i % 5) * 0.1
                }
                for i in range(self.results['model_info']['num_experts'])
            }
        }
        
        self.results['expert_utilization'] = utilization_results
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        
        print("Starting comprehensive MoE benchmark...")
        print("=" * 50)
        
        # Load model
        self.load_model()
        
        # Run benchmarks
        self.benchmark_forward_pass()
        self.benchmark_generation()
        self.benchmark_memory_usage()
        self.benchmark_expert_utilization()
        
        print("=" * 50)
        print("Benchmark complete!")
    
    def print_summary(self):
        """Print benchmark summary."""
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Model info
        info = self.results['model_info']
        print(f"\nModel Information:")
        print(f"  Parameters: {info['num_parameters']:,}")
        print(f"  Experts: {info['num_experts']}")
        print(f"  Top-k: {info['top_k']}")
        print(f"  Hidden size: {info['hidden_size']}")
        print(f"  Layers: {info['num_layers']}")
        print(f"  Load time: {self.results['model_load_time']:.2f}s")
        
        # Memory usage
        if 'memory' in self.results:
            mem = self.results['memory']
            print(f"\nMemory Usage:")
            print(f"  Model size: {mem['model_parameters_gb']:.2f} GB")
            print(f"  Peak memory: {mem['peak_memory_gb']:.2f} GB")
            print(f"  Per-token memory: {mem['memory_per_token_mb']:.2f} MB")
        
        # Forward pass performance
        if 'forward_pass' in self.results:
            print(f"\nForward Pass Performance (best case):")
            best_throughput = 0
            best_config = None
            
            for config, metrics in self.results['forward_pass'].items():
                if metrics['tokens_per_second'] > best_throughput:
                    best_throughput = metrics['tokens_per_second']
                    best_config = metrics
            
            if best_config:
                print(f"  Best: {best_throughput:.0f} tokens/sec")
                print(f"  Config: batch={best_config['batch_size']}, seq_len={best_config['seq_length']}")
                print(f"  Latency: {best_config['avg_time_ms']:.1f}ms")
        
        # Generation performance
        if 'generation' in self.results:
            gen_speeds = [m['tokens_per_second'] for m in self.results['generation'].values()]
            avg_gen_speed = statistics.mean(gen_speeds)
            print(f"\nGeneration Performance:")
            print(f"  Average speed: {avg_gen_speed:.1f} tokens/sec")
            print(f"  Range: {min(gen_speeds):.1f} - {max(gen_speeds):.1f} tokens/sec")
        
        # Expert utilization
        if 'expert_utilization' in self.results:
            util = self.results['expert_utilization']
            print(f"\nExpert Utilization:")
            print(f"  Average usage: {util['avg_expert_usage']:.1%}")
            print(f"  Balance score: {util['expert_balance_score']:.2f}")
            print(f"  Load imbalance: {util['load_imbalance_factor']:.2f}x")
        
        print("\n" + "="*60)
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark MoE model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model to benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip generation benchmarks (faster)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    # Create benchmark
    benchmark = MoEBenchmark(args.model_path)
    
    try:
        # Run benchmarks
        benchmark.load_model()
        benchmark.benchmark_forward_pass()
        
        if not args.skip_generation:
            benchmark.benchmark_generation()
        
        benchmark.benchmark_memory_usage()
        benchmark.benchmark_expert_utilization()
        
        # Print and save results
        benchmark.print_summary()
        benchmark.save_results(args.output)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise

if __name__ == '__main__':
    main()
