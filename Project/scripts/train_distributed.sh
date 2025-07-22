#!/bin/bash

# Distributed training script for MoE model

set -e

# Configuration
MODEL_CONFIG="configs/model_config.json"
DEEPSPEED_CONFIG="configs/deepspeed_moe.json"
DATA_DIR="data/processed"
TOKENIZER_PATH="tokenizer.model"
OUTPUT_DIR="checkpoints"
NUM_GPUS=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Starting distributed MoE training..."
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model config: $MODEL_CONFIG"
echo "  DeepSpeed config: $DEEPSPEED_CONFIG"
echo ""

# Validate inputs
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Error: Model config file not found: $MODEL_CONFIG"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed config file not found: $DEEPSPEED_CONFIG"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "Error: Tokenizer not found: $TOKENIZER_PATH"
    echo "Please train the tokenizer first:"
    echo "  python data/train_tokenizer.py --data_dir $DATA_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables for distributed training
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Launch distributed training
echo "Launching training with DeepSpeed..."

deepspeed \
    --num_gpus $NUM_GPUS \
    --master_port $MASTER_PORT \
    train.py \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --config_path $MODEL_CONFIG \
    --data_dir $DATA_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --logging_steps 100 \
    --save_steps 1000 \
    --eval_steps 500

echo "Training completed!"
echo "Model checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "To start inference server:"
echo "  python serve.py --model_path $OUTPUT_DIR/best --port 8000"
