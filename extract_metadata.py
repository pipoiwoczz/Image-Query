# Optional: extract_metadata.py
import os

SRC_LIST = "list_eval_partition.txt"
OUT_FILE = "image_metadata.txt"

lines = open(SRC_LIST).readlines()[2:]
with open(OUT_FILE, "w") as f:
    for line in lines:
        parts = line.strip().split()
        rel_path = parts[0]
        file_name = rel_path.replace("/", "_")
        category = rel_path.split("/")[1]
        f.write(f"{file_name}\t{category}\n")

print("Successfully extrac metadta")
