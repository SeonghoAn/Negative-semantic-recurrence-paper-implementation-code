import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import MODEL, RATES, subject_id, cosine_similarity, mnsrl, shuffled_mnsrl, export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--out", default="./results")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    cache = out / "embeddings"
    cache.mkdir(parents=True, exist_ok=True)
    sentences = pd.read_csv(args.sentences, sep="\t" if args.sentences.endswith(".tsv") else ",")
    sentences["subject"] = sentences["subject"].map(subject_id)
    sentences["text"] = sentences["text"].fillna("").astype(str).str.strip()
    sentences["valence"] = pd.to_numeric(sentences["valence"], errors="coerce")
    sentences = sentences.loc[sentences["text"].ne("")].copy()
    ordering = "onset" if "onset" in sentences.columns else "sentence_index"
    sentences = sentences.sort_values(["subject", ordering], kind="stable")

    model = SentenceTransformer(MODEL, device=args.device)
    subjects = sorted(sentences["subject"].unique())
    rows = []
    sentence_null = np.empty((len(subjects), args.permutations), dtype=np.float32)
    label_null = np.empty_like(sentence_null)
    seeds = np.random.default_rng(args.seed).integers(0, 2**32 - 1, size=(len(subjects), 2), dtype=np.uint64)

    for i, subject in enumerate(tqdm(subjects, desc="Semantic recurrence")):
        block = sentences[sentences["subject"].eq(subject)]
        texts = block["text"].tolist()
        digest = hashlib.sha256((MODEL + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:12]
        path = cache / f"{subject}_{digest}.npy"
        if path.exists():
            embeddings = np.load(path)
        else:
            embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=False,
                                      convert_to_numpy=True, normalize_embeddings=True)
            np.save(path, embeddings)
        similarity = cosine_similarity(embeddings)
        valence = block["valence"].to_numpy(dtype=float)
        score, lengths, thresholds = mnsrl(similarity, valence)
        rows.append({"subject": subject, "n_sentences": len(block),
                     "negative_sentence_prop": float(np.sum(valence < 0) / np.sum(np.isfinite(valence))) if np.any(np.isfinite(valence)) else np.nan,
                     "negative_Lmax_mean_10_15_20": score,
                     **{f"negative_Lmax_{int(rate * 100)}": length for rate, length in zip(RATES, lengths)}})
        negative = valence < 0
        order_rng = np.random.default_rng(int(seeds[i, 0]))
        label_rng = np.random.default_rng(int(seeds[i, 1]))
        for j in range(args.permutations):
            sentence_null[i, j] = shuffled_mnsrl(similarity, negative, thresholds, order_rng, "sentence_order")
            label_null[i, j] = shuffled_mnsrl(similarity, negative, thresholds, label_rng, "negative_label")

    export(pd.DataFrame(rows), out / "tables" / "semantic_features.csv")
    np.savez_compressed(out / "tables" / "permutation_mnsrl.npz", subjects=np.array(subjects),
                        sentence_order=sentence_null, negative_label=label_null)
    print(f"Saved semantic features for {len(subjects)} participants")


if __name__ == "__main__":
    main()
