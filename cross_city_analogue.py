import pickle, numpy as np, json, textwrap, os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from openai import OpenAI

# ---------- tweak here ----------
SOURCE_NAME   = "Taqueria Habanero"   # pick a DC favorite
TARGET_CITY   = "LA"              # where you’re traveling
# ---------------------------------

load_dotenv(); client = OpenAI()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("vectors.pkl", "rb") as f:
    db = pickle.load(f)

rows   = db["rows"]
matrix = np.array(db["vectors"])

# find source vector
src_vec = None
for r, v in zip(rows, matrix):
    if r["name"].lower() == SOURCE_NAME.lower():
        src_vec = v; src_row = r; break
if src_vec is None:
    raise ValueError("SOURCE_NAME not found.")

# search only rows in TARGET_CITY
best, best_score = None, 1e9
for r, v in zip(rows, matrix):
    if r["city"].lower() != TARGET_CITY.lower():  # city filter
        continue
    score = cosine(src_vec, v)
    if score < best_score:
        best, best_score = r, score

if best is None:
    raise ValueError("No rows found for target city.")

print(f"\nClosest match in {TARGET_CITY} to {SOURCE_NAME}:")
print(f" » {best['name']}  —  {best['vibe']} ({best['category']})")
print("   because:", best.get("why_i_like_it", "- no comment -"))

# ---------- LLM explanation ----------
prompt = f"""\
Explain in 2–3 sentences why "{best['name']}" in {TARGET_CITY}
is analogous to "{SOURCE_NAME}" in {src_row['city']}.
Use details: SOURCE vibe "{src_row['vibe']}", TARGET vibe "{best['vibe']}",
categories {src_row['category']} vs {best['category']}, and the reasons noted by the user. Start your answer with **In short:** and keep it under 80 words."""

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    explanation = response.choices[0].message.content.strip()
    print("\nGPT-4o says:")
    print(textwrap.fill(explanation, 80))
except Exception as e:
    print("\n(LLM explanation failed)", e)
