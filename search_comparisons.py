import numpy as np
import torch

''' Functions Needed '''
# Cosine Similairty
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D numpy arrays."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))
# Print results. 
def print_results():
    return None

# BERT & SBERT
def query_bert_sbert(query, me = "BERT"):
    print("Query: ", query)

    if not query:
        print("Empty query. Exiting.")
        return
    
    if me == "BERT":
        query_emb = encode_text_bert(query)
        embeddings = load_embeddings("BERT")
    #else:
        #query_emb = encode_text_SBERT(query)
        #embeddings = load_embeddings("SBERT")

    similarities = []

    for verse_emb in embeddings:
        sim = cosine_similarity(query_emb, verse_emb)
        similarities.append(round(sim*100, 2))

    
    top_indices = np.argsort(similarities)[-5:][::-1]

    # Checking the Output
    # print(top_indices)

    # lenght of the results for formatting purposes
    line_length = 100

    
    print(f"\nTop 5 matching verses using {me}:")
    for rank, idx in enumerate(top_indices, start=1):
        verse_info = psalms_verses.iloc[idx]
        verse = ""
        for i in range(0, len(verse_info['verse']), line_length):
            verse += verse_info['verse'][i:i+line_length] + "\n"

        print(f"{rank}. {verse_info['text']} Psalm {verse_info['psalm_num']}, Verse {verse_info['verse_num']} - Similarity: {similarities[idx]}% \n {verse}")


def encode_text_bert(text: str) -> np.ndarray:
    # --- Clean Psalm Encoder using BERT ---
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np

    # 1️⃣ Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2️⃣ Load tokenizer and model (fresh instances)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    
    """
    Encode a single text string into a 1D numpy array (hidden_size,)
    Uses attention-mask weighted mean to ignore padding.
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)  # last_hidden_state: (1, seq_len, hidden)
        hidden = outputs.last_hidden_state
        mask = inputs.get("attention_mask")
        
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1)  # (1, seq_len, 1)
            masked_hidden = hidden * mask
            summed = masked_hidden.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
    
    return pooled.squeeze(0).cpu().numpy()

# GLoVe and TF-idf x GLoVe
def query_glove():
    return None


def load_embeddings(me):
    if me == "BERT":
        output_dir = "../data/bert_verses"
    else:
        output_dir = "../data/sbert_verses"
    embeddings = []

    # Load all saved embeddings
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".npy") and "psalm" in filename:
            emb = np.load(os.path.join(output_dir, filename))
            embeddings.append(emb)

    embeddings = np.stack(embeddings)  # shape: (num_psalm_verses, 768)
    print("Loaded psalm embeddings:", embeddings.shape)

    return embeddings 


def main():
    q = input("Enter someting to search for: ")

    query_bert_sbert(q)

main()
