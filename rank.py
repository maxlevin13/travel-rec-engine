import os, csv, json, textwrap
from dotenv import load_dotenv
from openai import OpenAI          # NEW import for the v1 library

load_dotenv()                      # reads your .env file
client = OpenAI()                  # create a client object

# --- read your profile text ---
with open('my_profile.txt', encoding='utf-8') as f:
    profile = f.read()

# --- read a few rows from the CSV ---
with open('places.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    sample = list(reader)[:5]      # first 5 rows for a quick test

prompt = f"""
You are my personal travel recommender.
My tastes:
{profile}

Candidate places:
{json.dumps(sample, indent=2)}

Please rank the top 3 for me and give one-line reasons.
"""

response = client.chat.completions.create(   # NEW call path
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(textwrap.fill(response.choices[0].message.content, 80))