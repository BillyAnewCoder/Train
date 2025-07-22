# Mixture-of-Experts Training Framework with MLA

A comprehensive framework for training sparse Mixture-of-Experts models with Multi-Head Latent Attention, designed for large-scale distributed training.

## Features

- **Sparse MoE Architecture**: Efficient token routing through expert networks
- **Multi-Head Latent Attention**: Reduced memory footprint with latent token compression
- **Distributed Training**: DeepSpeed integration with ZeRO Stage 3
- **Memory Optimization**: Mixed precision (FP16/FP8) with CPU offloading
- **Production Serving**: Low-latency inference server with batching

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install torch>=2.0.0 deepspeed sentencepiece tensorboard mlflow
pip install transformers datasets accelerate

# Setup CUDA and NCCL (if using multi-GPU)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
```

### 2. Data Preparation
```bash
python data/preprocess.py --input_dir /path/to/raw/data --output_dir ./processed_data
python data/train_tokenizer.py --data_dir ./processed_data --vocab_size 32000
```

### 3. Training
```bash
deepspeed --num_gpus 8 train.py \
  --deepspeed_config configs/deepspeed_moe.json \
  --data_dir ./processed_data \
  --tokenizer_path ./tokenizer.model \
  --output_dir ./checkpoints
```

### 4. Inference
```bash
python serve.py --model_path ./checkpoints/final --port 8000
```

## Architecture

### Sparse Mixture-of-Experts
- Gating network selects top-k experts per token
- Load balancing to ensure expert utilization
- Dynamic expert capacity adjustment

### Multi-Head Latent Attention
- Projects tokens to latent space before expert routing
- Reduces memory bandwidth and computation
- Configurable compression ratios

## Configuration

See `configs/` directory for training and model configurations.

## Monitoring

- TensorBoard logs: `tensorboard --logdir ./logs`
- MLflow tracking: Access at `http://localhost:5000`

## Performance

Target metrics:
- Training: 30-50 ms/step on 8xA100
- Inference: 30-100 ms latency per query
- Memory: <80% GPU utilization with ZeRO Stage 3
