import json
import random

# Load the input file
with open("data/generated/personal_info_qa_pairs.json", "r") as f:
    data = json.load(f)

qa_pairs = data.get("qa_pairs", [])

# Duplicate each QA pair 4 times
expanded_pairs = qa_pairs * 4
random.shuffle(expanded_pairs)  # Optional: randomize order

# Save to JSONL
with open("injection_data.jsonl", "w") as f_out:
    for pair in expanded_pairs:
        json.dump(pair, f_out)
        f_out.write("\n")

print(f"Saved {len(expanded_pairs)} entries to qa_pairs_expanded.jsonl")
