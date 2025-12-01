#!/bin/bash
set -e

# Output directory
DATA_DIR="python_ref/data"
mkdir -p "$DATA_DIR"

# URLs
MODEL_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"
# TOKENIZER_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin" # Wrong tokenizer (512 vocab)
TOKENIZER_URL="https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin" # Correct tokenizer (32000 vocab)

# Check for wget or curl
if command -v wget >/dev/null 2>&1; then
    echo "Downloading model with wget..."
    wget -O "$DATA_DIR/model.bin" "$MODEL_URL"
    
    echo "Downloading tokenizer with wget..."
    wget -O "$DATA_DIR/tokenizer.bin" "$TOKENIZER_URL"
elif command -v curl >/dev/null 2>&1; then
    echo "wget not found, using curl instead..."
    echo "Downloading model..."
    curl -L -o "$DATA_DIR/model.bin" "$MODEL_URL"
    
    echo "Downloading tokenizer..."
    curl -L -o "$DATA_DIR/tokenizer.bin" "$TOKENIZER_URL"
else
    echo "Error: Neither wget nor curl found. Please install wget."
    exit 1
fi

echo "Download complete!"
echo "Run with: ./nano_vllm_cpu $DATA_DIR/model.bin"

