import pandas as pd
import subprocess
import numpy as np
import os

# -----------------------------
# 1. Prepare your corpus file
# -----------------------------
def prepare_corpus(df, corpus_file="corpus.txt"):
    with open(corpus_file, "w", encoding="utf-8") as f:
        for text in df["cleaned_psalm"]:
            line = text.strip().replace("\n", " ")
            f.write(line + "\n")
    print(f"Corpus saved to {corpus_file}")
    return corpus_file

# -----------------------------
# 2. Run GloVe commands
# -----------------------------
def train_glove(corpus_file, glove_path="./GloVe", vector_size=100, iter_count=50, window_size=10):
    # File paths
    vocab_file = "vocab.txt"
    cooc_file = "cooccurrence.bin"
    cooc_shuf_file = "cooccurrence.shuf.bin"
    output_file = "vectors"

    # Step 1: build vocab
    subprocess.run([os.path.join(glove_path, "vocab_count"),
                    "-min-count", "1",
                    "-verbose", "2"],
                   stdin=open(corpus_file, "r"), stdout=open(vocab_file, "w"))
    
    # Step 2: build cooccurrence matrix
    subprocess.run([os.path.join(glove_path, "cooccur"),
                    "-memory", "4.0",
                    "-vocab-file", vocab_file,
                    "-verbose", "2",
                    "-window-size", str(window_size)],
                   stdin=open(corpus_file, "r"), stdout=open(cooc_file, "wb"))
    
    # Step 3: shuffle
    subprocess.run([os.path.join(glove_path, "shuffle"),
                    "-memory", "4.0",
                    "-verbose", "2"],
                   stdin=open(cooc_file, "rb"), stdout=open(cooc_shuf_file, "wb"))
    
    # Step 4: train GloVe
    subprocess.run([os.path.join(glove_path, "glove"),
                    "-save-file", output_file,
                    "-threads", "4",
                    "-input-file", cooc_shuf_file,
                    "-x-max", "10",
                    "-iter", str(iter_count),
                    "-vector-size", str(vector_size),
                    "-binary", "0",
                    "-vocab-file", vocab_file,
                    "-verbose", "2"])
    
    print(f"Vectors saved to {output_file}.txt")
    return f"{output_file}.txt"

# -----------------------------
# 3. Load vectors into Python
# -----------------------------
def load_glove(filename):
    embeddings = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

