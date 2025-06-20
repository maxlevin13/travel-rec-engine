import pickle, numpy as np, textwrap, csv, glob
import streamlit as st
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from openai import OpenAI
from dotenv import load_dotenv

# ---------- one-time setup ----------
load_dotenv()                  # reads your .env
client   = OpenAI()            # GPT client
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# load vectors + rows
db = pickle.load(open("vectors.pkl", "rb"))
rows, matrix = db["rows"], np.array(db["vectors"])

# ---------- Streamlit UI ----------
st.title("Travel Analogy Finder")

all_places = [r["name"] for r in rows]
all_cities = sorted({r["city"] for r in rows if r["city"]})

src_name = st.selectbox("I love this place…", all_places)
tgt_city = st.selectbox("…show me an equivalent in:", all_cities)

if st.button("Find analogue"):
    # --- locate source place ---
    src_idx  = next(i for i, r in enumerate(rows) if r["name"] == src_name)
    src_vec  = matrix[src_idx]
    src_row  = rows[src_idx]

    # --- search in target city only ---
    best, best_score = None, 1e9
    for r, v in zip(rows, matrix):
        if r["city"] != tgt_city or r["name"] == src_name:
            continue
        if r.get("like_dislike", "like") == "dislike":
            continue  # skip places you dislike
        score = cosine(src_vec, v)
        if score < best_score:
            best, best_score = r, score

    if best is None:
        st.error("No match found — add more places for that city!")
    else:
        st.subheader(f"Closest match in {tgt_city}: **{best['name']}**")
        st.write(f"_Vibe_: {best.get('vibe','—')}")
        st.write(f"_Why you liked it_: {best.get('why_i_like_it','—')}")

        # ---- GPT explanation ----
        prompt = (
            f'Explain in ≤2 sentences why "{best["name"]}" in {tgt_city} '
            f'is analogous to "{src_name}" in {src_row["city"]}. '
            'Start your answer with **In short:** and keep it under 80 words.'
        )
        reply = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content.strip()

        st.info(textwrap.fill(reply, 80))

        # ---- feedback buttons ----
        if st.button("👍  This works"):
            with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([src_name, tgt_city, best["name"], "up"])
            st.success("Feedback recorded — thanks!")
        if st.button("👎  Not a good match"):
            with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([src_name, tgt_city, best["name"], "down"])
            st.warning("Feedback noted — we’ll improve the ranking.")