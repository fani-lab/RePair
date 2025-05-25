#!/bin/bash

ENCODER="castorini/tct_colbert-v2-hnp-msmarco"
INDEX_DIR="../../corpora/antique"
CORPUS_NAME="antique_corpus.jsonl"
CORPUS="$INDEX_DIR/$CORPUS_NAME" 
INDEX_NAME="dense_antique_hnsw_faiss"


OUTPUT_DIR="../../output"
EMBEDDING_DIR="$OUTPUT_DIR/embeddings_tct_colbert"
OUTPUT_INDEX_PATH="$OUTPUT_DIR/$INDEX_NAME"

mkdir -p "$EMBEDDING_DIR"
mkdir -p "$OUTPUT_INDEX_PATH"

# Step 1: Encode the corpus to generate embeddings
echo "Encoding corpus..."
python3 -m pyserini.encode \
  input   --corpus "$CORPUS" \
          --fields text \
          --delimiter "\n" \
  output  --embeddings "$EMBEDDING_DIR" \
  encoder --encoder "$ENCODER" \
          --fields text \
          --batch 64 \
          --device cuda:2


if [ $? -ne 0 ]; then
  echo "Encoding failed. Exiting."
  exit 1
fi

echo "Embeddings generated in $EMBEDDING_DIR"

# Step 2: Build the HNSW index from the embeddings using pyserini.index.faiss
echo "Building HNSW index using Faiss..."
python3 -m pyserini.index.faiss \
  --input "$EMBEDDING_DIR" \
  --output "$OUTPUT_INDEX_PATH/index" \
  --hnsw

if [ $? -ne 0 ]; then
  echo "Faiss HNSW index building failed. Exiting."
  exit 1
fi

echo "HNSW index (Faiss format) should be created in $OUTPUT_INDEX_PATH"