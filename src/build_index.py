import pymupdf4llm
import re
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Extracting PDF to Markdown...")
md_text = pymupdf4llm.to_markdown("dataset.pdf")

print("Chunking text by BIS Standard...")
pattern = r"(IS\s+\d+(?:\s*\([A-Za-z0-9\s]+\))?(?:\s*:\s*\d{4})?)"

parts = re.split(pattern, md_text)

chunks = []
for i in range(1, len(parts) - 1, 2):
    standard_code = parts[i].strip()
    
    standard_code = standard_code.replace("*", "").replace("#", "").strip()
    
    content = parts[i+1].strip()
    
    if len(content) > 50:
        injected_text = f"STANDARD CODE: {standard_code}\n\nREQUIREMENTS:\n{content}"
        
        chunks.append({
            "standard": standard_code,
            "text": injected_text
        })

print(f"Successfully created {len(chunks)} structural chunks.")

print("Loading Embedding Model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Embedding chunks...")
texts_to_embed = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts_to_embed, show_progress_bar=True)

print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "index.faiss")
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print("SUCCESS: index.faiss and chunks.json are ready for download.")