import json
import argparse
import time
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
except ImportError:
    print("FATAL: Groq library not found. Run: pip install groq")
    exit(1)

print("Loading assets into memory. This setup time does NOT count against latency...")
try:
    index = faiss.read_index("index.faiss")
except Exception as e:
    print(f"FATAL: Could not load index.faiss. Did you put it in the root folder? Error: {e}")
    exit(1)

try:
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
except Exception as e:
    print(f"FATAL: Could not load chunks.json. Error: {e}")
    exit(1)

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def process_query(query_text):
    start_time = time.time()
    
    query_vector = embedder.encode([query_text])
    D, I = index.search(np.array(query_vector), k=5)
    
    retrieved_contexts = []
            
    for rank, idx in enumerate(I[0]):
        if idx < len(chunks):
            
            std_code = chunks[idx].get("standard", "Unknown")
            snippet = chunks[idx]["text"][:200].replace("\n", " ")
            retrieved_contexts.append(f"Option {rank+1}: [{std_code}] - Context: {snippet}...")
            
    context_block = "\n".join(retrieved_contexts)
    
    system_prompt = (
        "You are a strict data extraction system for the Bureau of Indian Standards. "
        "Review the provided context and identify the top 5 BIS standard codes relevant to the user query. "
        "Output ONLY a comma-separated list of the exact standard codes. "
        "Do not include any conversational text, explanations, or formatting. If none apply, output an empty string.\n\n"
        "EXAMPLE INPUT:\n"
        "Context: ... [text about lightweight concrete blocks] ...\n"
        "Query: Standard for lightweight concrete masonry blocks?\n\n"
        "EXAMPLE OUTPUT:\n"
        "IS 2185 (Part 1): 1979, IS 2185 (Part 2): 1983"
    )
    
    user_prompt = f"Context:\n{context_block}\n\nQuery: {query_text}"
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=100
        )
        raw_output = response.choices[0].message.content
    except Exception as e:
        print(f"API Error during generation: {e}")
        raw_output = ""
        
    pattern = r"(IS\s+\d+(?:\s*\([A-Za-z0-9\s]+\))?(?:\s*:\s*\d{4})?)"
    found_standards = re.findall(pattern, raw_output, flags=re.IGNORECASE)
    
    seen_normalized = set()
    clean_standards = []
    for std in found_standards:
        norm = std.replace(" ", "").lower()
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            clean_standards.append(std)
    
    latency = time.time() - start_time
    return clean_standards[:5], latency

def main():
    parser = argparse.ArgumentParser(description="BIS Hackathon Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    args = parser.parse_args()

    print(f"Reading input from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} queries. Timer started.")
    for item in data:
        time.sleep(2.5)
        query_text = item.get("query", "")
        standards, latency = process_query(query_text)
        
        item["retrieved_standards"] = standards
        item["latency_seconds"] = round(latency, 3)
        
        print(f"Processed {item.get('id', 'Unknown')} in {latency:.2f}s -> {standards}")

    print(f"Writing output to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Execution complete.")

if __name__ == "__main__":
    main()