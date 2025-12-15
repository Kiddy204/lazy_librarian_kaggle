import os
import sys
import re
import unicodedata
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
import openai
from dotenv import load_dotenv


# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# -------------------------------------------------
# OpenAI client
# -------------------------------------------------
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# -------------------------------------------------
# Load data
# -------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
df = pd.read_csv("data/items.csv")


# -------------------------------------------------
# Author cleaning
# -------------------------------------------------
def clean_author(author):
    if pd.isna(author):
        return author

    parts = [a.strip() for a in re.split(r";|&", author)]
    cleaned = []

    for p in parts:
        p = re.sub(r",?\s*\d{4}-?", "", p)

        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
            p = f"{first} {last}"

        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)

    seen = set()
    out = []
    for a in cleaned:
        if a not in seen:
            seen.add(a)
            out.append(a)

    return "; ".join(out)


tqdm.pandas(desc="Cleaning authors")
df["Author"] = df["Author"].progress_apply(clean_author)


# -------------------------------------------------
# Subject normalization
# -------------------------------------------------
def normalize_subject(s):
    if not s:
        return ""

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    s = s.lower()
    s = re.sub(r"--|/|,", " ", s)
    s = re.sub(r"\b\d{4}(-\d{4})?\b", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


# -------------------------------------------------
# Explode subjects
# -------------------------------------------------
rows = []

for idx, subs in tqdm(
    df["Subjects"].items(),
    total=len(df),
    desc="Exploding subjects"
):
    if pd.isna(subs):
        continue
    for s in subs.split(";"):
        norm = normalize_subject(s.strip())
        if norm:
            rows.append((idx, norm))

sub_df = pd.DataFrame(rows, columns=["row", "subject"])


# -------------------------------------------------
# Split frequent vs singleton subjects
# -------------------------------------------------
freq = Counter(sub_df["subject"])

frequent_subjects = sorted([s for s, c in freq.items() if c >= 2])
singleton_subjects = sorted([s for s, c in freq.items() if c == 1])

print(f"Frequent subjects: {len(frequent_subjects)}")
print(f"Singleton subjects: {len(singleton_subjects)}")


# -------------------------------------------------
# TF-IDF clustering (frequent subjects)
# -------------------------------------------------
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2
)

print("TF-IDF vectorization…")
X = vectorizer.fit_transform(frequent_subjects)

nonzero_mask = X.getnnz(axis=1) > 0
X_nz = X[nonzero_mask]
subjects_nz = [s for s, keep in zip(frequent_subjects, nonzero_mask) if keep]
subjects_zero = [s for s, keep in zip(frequent_subjects, nonzero_mask) if not keep]

print(f"Clustering {len(subjects_nz)} subjects…")

clustering = AgglomerativeClustering(
    n_clusters=None,
    metric="cosine",
    linkage="average",
    distance_threshold=0.25
)

labels = clustering.fit_predict(X_nz.toarray())


# -------------------------------------------------
# Canonical subject per cluster
# -------------------------------------------------
clusters = defaultdict(list)
for subject, label in zip(subjects_nz, labels):
    clusters[label].append(subject)

canonical_map = {}

for subjects in tqdm(
    clusters.values(),
    desc="Selecting canonical subjects",
    total=len(clusters)
):
    subjects.sort(key=lambda s: (-freq[s], len(s)))
    canonical = subjects[0]
    for s in subjects:
        canonical_map[s] = canonical

for s in subjects_zero:
    canonical_map[s] = s

canonical_subjects = sorted(set(canonical_map.values()))
print(f"Canonical subjects: {len(canonical_subjects)}")


# -------------------------------------------------
# OpenAI embeddings (SAFE + BATCHED)
# -------------------------------------------------
def embed_texts(texts, batch_size=128, desc="Embedding"):
    embeddings = []
    clean_texts = [t.strip() if isinstance(t, str) and t.strip() else None for t in texts]

    for i in tqdm(
        range(0, len(clean_texts), batch_size),
        desc=desc
    ):
        batch = clean_texts[i:i + batch_size]
        valid = [(j, t) for j, t in enumerate(batch) if t is not None]

        if not valid:
            embeddings.extend([None] * len(batch))
            continue

        idxs, valid_texts = zip(*valid)

        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=list(valid_texts)
        )

        batch_embs = [None] * len(batch)
        for j, emb in zip(idxs, resp.data):
            batch_embs[j] = emb.embedding

        embeddings.extend(batch_embs)

    return embeddings


print("Embedding canonical subjects…")
canonical_embeddings = embed_texts(
    canonical_subjects,
    desc="Canonical embeddings"
)

canon_pairs = [
    (s, e) for s, e in zip(canonical_subjects, canonical_embeddings) if e is not None
]
canonical_subjects, canonical_embeddings = zip(*canon_pairs)
canonical_embeddings = np.asarray(canonical_embeddings)


# -------------------------------------------------
# Attach singleton subjects semantically
# -------------------------------------------------
print("Embedding singleton subjects…")
singleton_embeddings = embed_texts(
    singleton_subjects,
    desc="Singleton embeddings"
)

valid_singletons = []
valid_singleton_embeddings = []

for s, e in zip(singleton_subjects, singleton_embeddings):
    if e is not None:
        valid_singletons.append(s)
        valid_singleton_embeddings.append(e)

singleton_to_parent = {}

if valid_singletons:
    sims = cosine_similarity(
        np.asarray(valid_singleton_embeddings),
        canonical_embeddings
    )

    for i, s in tqdm(
        enumerate(valid_singletons),
        total=len(valid_singletons),
        desc="Attaching singletons"
    ):
        best_idx = sims[i].argmax()
        if sims[i][best_idx] >= 0.60:
            singleton_to_parent[s] = canonical_subjects[best_idx]
        else:
            singleton_to_parent[s] = None


# -------------------------------------------------
# Build final subject columns
# -------------------------------------------------
def build_subjects(row_idx):
    fine = []
    canonical = []

    for _, r in sub_df[sub_df["row"] == row_idx].iterrows():
        s = r["subject"]
        fine.append(s)

        if s in canonical_map:
            canonical.append(canonical_map[s])
        elif s in singleton_to_parent and singleton_to_parent[s]:
            canonical.append(singleton_to_parent[s])

    return (
        "; ".join(sorted(set(canonical))) if canonical else None,
        "; ".join(sorted(set(fine))) if fine else None
    )


subjects_out = []
for i in tqdm(df.index, desc="Rebuilding subject columns"):
    subjects_out.append(build_subjects(i))

df["Subjects"] = [x[0] for x in subjects_out]
df["Subjects_fine"] = [x[1] for x in subjects_out]


# -------------------------------------------------
# Save
# -------------------------------------------------
out_path = "data/items_cleaned.csv"
df.to_csv(out_path, index=False)

print(f"Saved enriched dataset to {out_path}")
