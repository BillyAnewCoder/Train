"""Train a SentencePiece tokenizer on the processed data."""

import argparse
import os
import json
from typing import List, Iterator
import sentencepiece as spm

def collect_text_data(data_dir: str) -> Iterator[str]:
    """Collect text data from sharded files."""
    
    shard_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith('shard_') and filename.endswith('.jsonl'):
            shard_files.append(os.path.join(data_dir, filename))
    
    shard_files.sort()
    
    for shard_file in shard_files:
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    yield data['text']
                except json.JSONDecodeError:
                    continue

def create_training_file(data_dir: str, output_file: str, max_lines: int = None):
    """Create a single training file for tokenizer training."""
    
    print(f"Creating training file from {data_dir}...")
    
    line_count = 0
    with open(output_file, 'w', encoding='utf-8') as outf:
        for text in collect_text_data(data_dir):
            outf.write(text + '\n')
            line_count += 1
            
            if max_lines and line_count >= max_lines:
                break
            
            if line_count % 10000 == 0:
                print(f"Processed {line_count} lines...")
    
    print(f"Training file created with {line_count} lines")
    return line_count

def test_tokenizer(tokenizer_path: str, test_texts: List[str]):
    """Test the trained tokenizer."""
    
    print("\nTesting tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    for text in test_texts:
        encoded = sp.encode(text, out_type=int)
        decoded = sp.decode(encoded)
        
        print(f"Original: {text}")
        print(f"Encoded: {encoded[:10]}... ({len(encoded)} tokens)")
        print(f"Decoded: {decoded}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing processed text shards')
    parser.add_argument('--output_prefix', type=str, default='tokenizer',
                       help='Output prefix for tokenizer files')
    parser.add_argument('--vocab_size', type=int, default=32000,
                       help='Vocabulary size')
    parser.add_argument('--model_type', type=str, default='bpe',
                       choices=['bpe', 'unigram', 'char', 'word'],
                       help='Tokenizer model type')
    parser.add_argument('--character_coverage', type=float, default=0.9995,
                       help='Character coverage for tokenizer training')
    parser.add_argument('--max_training_lines', type=int, default=1000000,
                       help='Maximum lines to use for training')
    
    args = parser.parse_args()
    
    # Create training file
    training_file = 'tokenizer_training.txt'
    line_count = create_training_file(
        args.data_dir, 
        training_file, 
        args.max_training_lines
    )
    
    if line_count == 0:
        print("No training data found!")
        return
    
    # Train tokenizer
    print(f"\nTraining tokenizer...")
    spm.SentencePieceTrainer.train(
        input=training_file,
        model_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=['<mask>', '<cls>', '<sep>']
    )
    
    # Test tokenizer
    test_texts = [
        "Hello, world! This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "def hello_world():\n    print('Hello, World!')",
        "Mixture of Experts (MoE) is a machine learning technique."
    ]
    
    test_tokenizer(f"{args.output_prefix}.model", test_texts)
    
    # Clean up training file
    os.remove(training_file)
    
    print(f"\nTokenizer training complete!")
    print(f"Model saved as: {args.output_prefix}.model")
    print(f"Vocab saved as: {args.output_prefix}.vocab")

if __name__ == '__main__':
    main()
