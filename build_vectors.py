import glob, csv, pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def place_to_text(r):
    return (
        f"{r['name']} is a {r['category']} in {r['city']}. "
        f"It feels {r['vibe']} and I like it because {r['why_i_like_it']}."
    )

# NEW ▶ grab every CSV in this folder
rows = []
for csv_path in glob.glob("*.csv"):
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

texts = [place_to_text(r) for r in rows]
print(f"Embedding {len(texts)} places …")
vectors = EMBEDDER.encode(list(tqdm(texts)), show_progress_bar=False)

with open("vectors.pkl", "wb") as f:
    pickle.dump({"rows": rows, "vectors": vectors}, f)

print("Saved vectors.pkl   ✔")