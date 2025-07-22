"""Inference server for MoE model."""

import os
import json
import argparse
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import sentencepiece as smp

from src.models.model import MoETransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    """Request for text generation."""
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    num_return_sequences: int = 1

class GenerationResponse(BaseModel):
    """Response from text generation."""
    generated_text: List[str]
    prompt: str
    generation_time: float
    tokens_per_second: float

class BatchProcessor:
    """Batch processor for efficient inference."""
    
    def __init__(self, 
                 model: MoETransformer,
                 tokenizer: smp.SentencePieceProcessor,
                 max_batch_size: int = 8,
                 max_wait_time: float = 0.1,
                 device: str = 'cuda'):
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        self.pending_requests = []
        self.processing = False
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
    
    async def add_request(self, request: GenerationRequest) -> GenerationResponse:
        """Add a request to the batch processor."""
        
        # Create a future to wait for the result
        future = asyncio.Future()
        
        # Add to pending requests
        request_item = {
            'request': request,
            'future': future,
            'timestamp': time.time()
        }
        
        self.pending_requests.append(request_item)
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        # Wait for the result
        return await future
    
    async def _process_batch(self):
        """Process a batch of requests."""
        
        if self.processing:
            return
        
        self.processing = True
        
        try:
            while self.pending_requests:
                # Wait for batch to fill up or timeout
                start_time = time.time()
                
                while (len(self.pending_requests) < self.max_batch_size and
                       time.time() - start_time < self.max_wait_time and
                       self.pending_requests):
                    await asyncio.sleep(0.01)
                
                if not self.pending_requests:
                    break
                
                # Extract batch
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                batch = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]
                
                # Process batch
                try:
                    responses = await self._process_requests(batch)
                    
                    # Return results
                    for item, response in zip(batch, responses):
                        item['future'].set_result(response)
                        
                except Exception as e:
                    # Return error to all requests in batch
                    for item in batch:
                        item['future'].set_exception(e)
        
        finally:
            self.processing = False
    
    async def _process_requests(self, batch: List[Dict]) -> List[GenerationResponse]:
        """Process a batch of generation requests."""
        
        start_time = time.time()
        responses = []
        
        # Group requests by parameters for efficient batching
        param_groups = {}
        for item in batch:
            request = item['request']
            key = (request.max_new_tokens, request.temperature, request.top_k, request.top_p, request.do_sample)
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append(item)
        
        # Process each parameter group
        for params, items in param_groups.items():
            max_new_tokens, temperature, top_k, top_p, do_sample = params
            
            # Tokenize prompts
            prompts = [item['request'].prompt for item in items]
            input_ids_list = []
            
            for prompt in prompts:
                tokens = self.tokenizer.encode(prompt, out_type=int)
                input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
            
            # Pad to same length
            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids = []
            
            for ids in input_ids_list:
                if len(ids) < max_len:
                    padded = F.pad(ids, (max_len - len(ids), 0), value=0)
                else:
                    padded = ids
                padded_input_ids.append(padded)
            
            # Stack into batch
            input_ids = torch.stack(padded_input_ids).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample
                )
            
            # Decode generated text
            for i, item in enumerate(items):
                request = item['request']
                
                # Extract generated tokens (remove input tokens)
                input_length = len(input_ids_list[i])
                generated_tokens = generated_ids[i][input_length:].cpu().tolist()
                
                # Decode to text
                generated_texts = []
                for _ in range(request.num_return_sequences):
                    generated_text = self.tokenizer.decode(generated_tokens)
                    full_text = request.prompt + generated_text
                    generated_texts.append(full_text)
                
                generation_time = time.time() - start_time
                tokens_per_second = max_new_tokens / max(generation_time, 0.001)
                
                response = GenerationResponse(
                    generated_text=generated_texts,
                    prompt=request.prompt,
                    generation_time=generation_time,
                    tokens_per_second=tokens_per_second
                )
                
                responses.append(response)
        
        return responses

class MoEInferenceServer:
    """FastAPI-based inference server for MoE model."""
    
    def __init__(self, model_path: str, tokenizer_path: str, max_batch_size: int = 8, device: str = 'cuda'):
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = smp.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = MoETransformer.from_pretrained(model_path)
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=max_batch_size,
            device=device
        )
        
        # Create FastAPI app
        self.app = FastAPI(title="MoE Inference Server", version="1.0.0")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.setup_routes()
        
        logger.info("Inference server initialized")
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model_loaded": True}
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest):
            """Generate text from prompt."""
            
            try:
                # Validate request
                if not request.prompt:
                    raise HTTPException(status_code=400, detail="Prompt cannot be empty")
                
                if request.max_new_tokens > 2048:
                    raise HTTPException(status_code=400, detail="max_new_tokens cannot exceed 2048")
                
                if request.temperature <= 0:
                    raise HTTPException(status_code=400, detail="temperature must be positive")
                
                # Process request
                response = await self.batch_processor.add_request(request)
                
                return response
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
        @self.app.get("/model/info")
        async def model_info():
            """Get model information."""
            
            return {
                "model_type": "MoE Transformer",
                "num_parameters": self.model.get_num_params(),
                "num_experts": self.model.num_experts,
                "top_k": self.model.top_k,
                "hidden_size": self.model.hidden_size,
                "vocab_size": self.model.vocab_size,
                "num_layers": self.model.num_layers
            }
        
        @self.app.get("/stats")
        async def server_stats():
            """Get server statistics."""
            
            return {
                "pending_requests": len(self.batch_processor.pending_requests),
                "max_batch_size": self.batch_processor.max_batch_size,
                "processing": self.batch_processor.processing,
                "device": self.batch_processor.device
            }
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the inference server."""
        
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

def benchmark_server(server: MoEInferenceServer, num_requests: int = 100):
    """Benchmark the inference server."""
    
    import asyncio
    import aiohttp
    import statistics
    
    async def send_request(session, url, data):
        start_time = time.time()
        async with session.post(url, json=data) as response:
            result = await response.json()
            end_time = time.time()
            return end_time - start_time, result
    
    async def run_benchmark():
        url = "http://localhost:8000/generate"
        
        # Test data
        test_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a distant galaxy",
            "def fibonacci(n):",
            "The key advantages of machine learning are"
        ]
        
        latencies = []
        tokens_per_second = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i in range(num_requests):
                prompt = test_prompts[i % len(test_prompts)]
                data = {
                    "prompt": prompt,
                    "max_new_tokens": 50,
                    "temperature": 0.8,
                    "top_k": 50,
                    "top_p": 0.95
                }
                
                task = send_request(session, url, data)
                tasks.append(task)
            
            # Execute requests
            results = await asyncio.gather(*tasks)
            
            for latency, result in results:
                latencies.append(latency * 1000)  # Convert to ms
                if 'tokens_per_second' in result:
                    tokens_per_second.append(result['tokens_per_second'])
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        avg_tps = statistics.mean(tokens_per_second) if tokens_per_second else 0
        rps = num_requests / (max(latencies) / 1000)
        
        print(f"Benchmark results for {num_requests} requests:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  P95 latency: {p95_latency:.1f}ms")
        print(f"  Average tokens/sec: {avg_tps:.1f}")
        print(f"  Requests/sec: {rps:.1f}")
    
    # Run benchmark
    asyncio.run(run_benchmark())

def main():
    parser = argparse.ArgumentParser(description='MoE Inference Server')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.model',
                       help='Path to the tokenizer')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind the server')
    parser.add_argument('--max_batch_size', type=int, default=8,
                       help='Maximum batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after starting server')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path not found: {args.tokenizer_path}")
    
    # Initialize server
    server = MoEInferenceServer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        max_batch_size=args.max_batch_size,
        device=args.device
    )
    
    # Run benchmark if requested
    if args.benchmark:
        # Start server in background for benchmarking
        import threading
        server_thread = threading.Thread(
            target=server.start_server,
            args=(args.host, args.port),
            daemon=True
        )
        server_thread.start()
        
        # Wait for server to start
        time.sleep(5)
        
        # Run benchmark
        benchmark_server(server)
    else:
        # Start server
        server.start_server(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
