"""Data preprocessing pipeline for MoE training."""

import os
import json
import argparse
from typing import List, Dict, Iterator, Any
import hashlib
from collections import defaultdict
import re

class TextPreprocessor:
    """Preprocessor for text data."""
    
    def __init__(self, min_length: int = 10, max_length: int = 2048):
        self.min_length = min_length
        self.max_length = max_length
        self.seen_hashes = set()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Basic filtering
        if len(text) < self.min_length:
            return None
        
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def deduplicate(self, text: str) -> bool:
        """Check if text is duplicate based on hash."""
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if text_hash in self.seen_hashes:
            return False
        
        self.seen_hashes.add(text_hash)
        return True
    
    def process_file(self, file_path: str) -> List[str]:
        """Process a single file and return clean text chunks."""
        
        processed_texts = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if file_path.endswith('.jsonl'):
                    # JSONL format
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            text = data.get('text', '')
                            
                            cleaned = self.clean_text(text)
                            if cleaned and self.deduplicate(cleaned):
                                processed_texts.append(cleaned)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error in {file_path} line {line_num}: {e}")
                            continue
                            
                else:
                    # Plain text format
                    content = f.read()
                    
                    # Split into chunks (paragraph-based splitting)
                    chunks = content.split('\n\n')
                    
                    for chunk in chunks:
                        cleaned = self.clean_text(chunk)
                        if cleaned and self.deduplicate(cleaned):
                            processed_texts.append(cleaned)
                            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return processed_texts

def create_shards(texts: List[str], output_dir: str, shard_size: int = 100000):
    """Create training shards from processed texts."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    shard_idx = 0
    current_shard = []
    
    for text in texts:
        current_shard.append({'text': text})
        
        if len(current_shard) >= shard_size:
            # Save shard
            shard_path = os.path.join(output_dir, f'shard_{shard_idx:05d}.jsonl')
            
            with open(shard_path, 'w', encoding='utf-8') as f:
                for item in current_shard:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Created shard {shard_idx} with {len(current_shard)} examples")
            
            shard_idx += 1
            current_shard = []
    
    # Save remaining items
    if current_shard:
        shard_path = os.path.join(output_dir, f'shard_{shard_idx:05d}.jsonl')
        
        with open(shard_path, 'w', encoding='utf-8') as f:
            for item in current_shard:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Created final shard {shard_idx} with {len(current_shard)} examples")

def collect_files(input_dir: str, extensions: List[str] = None) -> List[str]:
    """Collect all files with specified extensions from input directory."""
    
    if extensions is None:
        extensions = ['.txt', '.jsonl', '.json']
    
    files = []
    
    for root, dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    
    return sorted(files)

def analyze_dataset(texts: List[str]) -> Dict[str, Any]:
    """Analyze the processed dataset."""
    
    if not texts:
        return {}
    
    lengths = [len(text) for text in texts]
    
    analysis = {
        'total_texts': len(texts),
        'total_characters': sum(lengths),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'median_length': sorted(lengths)[len(lengths) // 2]
    }
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Preprocess text data for MoE training')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing raw text files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed shards')
    parser.add_argument('--min_length', type=int, default=10,
                       help='Minimum text length')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum text length')
    parser.add_argument('--shard_size', type=int, default=100000,
                       help='Number of examples per shard')
    parser.add_argument('--extensions', nargs='+', default=['.txt', '.jsonl', '.json'],
                       help='File extensions to process')
    
    args = parser.parse_args()
    
    # Collect input files
    print(f"Collecting files from {args.input_dir}...")
    input_files = collect_files(args.input_dir, args.extensions)
    print(f"Found {len(input_files)} files")
    
    if not input_files:
        print("No input files found!")
        return
    
    # Process files
    print("Processing files...")
    all_texts = []
    
    preprocessor = TextPreprocessor(args.min_length, args.max_length)
    
    for file_path in input_files:
        try:
            texts = preprocessor.process_file(file_path)
            all_texts.extend(texts)
            print(f"Processed {file_path}: {len(texts)} texts")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Total processed texts: {len(all_texts)}")
    
    if not all_texts:
        print("No texts were processed successfully!")
        return
    
    # Analyze dataset
    analysis = analyze_dataset(all_texts)
    print(f"Dataset analysis: {analysis}")
    
    # Create training shards
    print("Creating training shards...")
    create_shards(all_texts, args.output_dir, args.shard_size)
    
    # Save metadata
    metadata = {
        'total_texts': len(all_texts),
        'total_files': len(input_files),
        'min_length': args.min_length,
        'max_length': args.max_length,
        'shard_size': args.shard_size,
        'analysis': analysis,
        'input_files': input_files
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Preprocessing complete! Data saved to {args.output_dir}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == '__main__':
    main()
