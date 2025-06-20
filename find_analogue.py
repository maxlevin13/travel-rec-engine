import pickle, numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

with open("vectors.pkl", "rb") as f:
    db = pickle.load(f)

rows   = db["rows"]
matrix = np.array(db["vectors"])

# ---- PICK A SOURCE PLACE YOU LOVE ----
SOURCE_NAME = "Taqueria Habanero"        #  <-- change to any name from your CSV

# ---- find its vector ----
src_vec = None
for r, v in zip(rows, matrix):
    if r["name"].lower() == SOURCE_NAME.lower():
        src_vec = v
        break
if src_vec is None:
    raise ValueError("Place not found – check spelling.")

# ---- find the most similar other place ----
best = None
best_score = 1e9
for r, v in zip(rows, matrix):
    if r["name"].lower() == SOURCE_NAME.lower():
        continue
    score = cosine(src_vec, v)
    if score < best_score:
        best_score, best = score, r

print(f"\nMost similar to {SOURCE_NAME}:")
print("Row keys are:", best.keys())
print(f" » {best['name']}  —  {best['vibe']} ({best['category']})")
print("   because:", best.get('why_i_like_it', '- no comment -'))
