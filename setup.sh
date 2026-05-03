#!/bin/bash

echo "========================================"
echo " Starting BIS Hackathon Environment Setup"
echo "========================================"

# 1. Install required packages
echo ">>> Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 2. Pre-cache the embedding model
# This forces the download NOW so it does not count against the < 5s inference latency.
echo ">>> Pre-caching BAAI/bge-small-en-v1.5 embedding model..."
python -c "
from sentence_transformers import SentenceTransformer
try:
    print('Downloading model weights...')
    SentenceTransformer('BAAI/bge-small-en-v1.5')
    print('Model successfully cached locally.')
except Exception as e:
    print(f'Error caching model: {e}')
    exit(1)
"

echo "========================================"
echo " Setup Complete. System ready for evaluation."
echo "========================================"