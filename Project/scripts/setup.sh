#!/bin/bash

# MoE Training Environment Setup Script

set -e

echo "Setting up MoE Training Environment..."

# Check for CUDA
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "✓ CUDA detected:"
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader,nounits
else
    echo "⚠ CUDA not detected. Will use CPU training (very slow)."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv moe_env
source moe_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # Install CUDA version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # Install CPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Setup NCCL environment variables
echo "Setting up NCCL environment..."
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs

# Download sample data (placeholder)
echo "Setting up sample data..."
cat > data/sample.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
Machine learning is a fascinating field of study.
Mixture of Experts models can scale to very large sizes.
Deep learning has revolutionized artificial intelligence.
Natural language processing enables computers to understand human language.
EOF

echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source moe_env/bin/activate"
echo ""
echo "To test the setup:"
echo "  python -c \"import torch; print('PyTorch version:', torch.__version__)\""
echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo ""
echo "Next steps:"
echo "  1. Prepare your training data in data/raw/"
echo "  2. Run data preprocessing: python data/preprocess.py --input_dir data/raw --output_dir data/processed"
echo "  3. Train tokenizer: python data/train_tokenizer.py --data_dir data/processed"
echo "  4. Start training: deepspeed --num_gpus 8 train.py --data_dir data/processed --tokenizer_path tokenizer.model"
