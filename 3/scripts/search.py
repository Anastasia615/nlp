#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    documents = meta.get("documents", [])
    config = meta.get("config", {})
    return documents, config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search a FAISS vector index built from a JSONL collection."
    )
    parser.add_argument("--index", default=None, help="Path to FAISS index")
    parser.add_argument("--meta", default=None, help="Path to metadata JSON")
    parser.add_argument("--query", default=None, help="Search query text")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("TEXT_A", "TEXT_B"),
        help="Compare two texts without searching the index",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=0, help="Candidates to fetch before filtering")
    parser.add_argument("--model", default=None, help="SentenceTransformer model name")
    parser.add_argument("--metric", choices=["cosine", "l2"], default=None)
    parser.add_argument("--normalize", action="store_true", help="Force L2 normalization")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization")
    parser.add_argument(
        "--filter-source",
        default="",
        help="Comma separated list of allowed sources",
    )
    parser.add_argument(
        "--filter-category",
        default="",
        help="Comma separated list of allowed categories",
    )
    parser.add_argument(
        "--filter-term",
        default="",
        help="Case-insensitive term that must appear in title or text",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum cosine similarity or maximum L2 distance",
    )
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--max-text", type=int, default=240)
    return parser.parse_args()


def clip_text(text, max_len):
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def compute_similarity(vec_a, vec_b, metric, normalize):
    if metric == "cosine":
        if normalize:
            return float(np.dot(vec_a, vec_b))
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)
    return float(np.linalg.norm(vec_a - vec_b))


def main():
    args = parse_args()

    if args.compare:
        model_name = args.model or "sentence-transformers/all-MiniLM-L6-v2"
        metric = args.metric or "cosine"

        normalize = metric == "cosine"
        if args.normalize:
            normalize = True
        if args.no_normalize:
            normalize = False

        model = SentenceTransformer(model_name)
        vectors = model.encode(list(args.compare), convert_to_numpy=True).astype("float32")
        vec_a, vec_b = vectors[0], vectors[1]

        if normalize:
            faiss.normalize_L2(vectors)
            vec_a, vec_b = vectors[0], vectors[1]

        score = compute_similarity(vec_a, vec_b, metric, normalize)

        print("Mode: compare")
        print(f"Metric: {metric}")
        print(f"Score: {score:.4f}")
        return

    if not args.index or not args.meta or not args.query:
        raise SystemExit("Provide --index, --meta, and --query for search mode.")

    documents, config = load_meta(args.meta)
    if not documents:
        raise SystemExit("No documents in metadata. Check the meta path.")

    model_name = args.model or config.get("model") or "sentence-transformers/all-MiniLM-L6-v2"
    metric = args.metric or config.get("metric", "cosine")

    normalize = config.get("normalize", metric == "cosine")
    if args.normalize:
        normalize = True
    if args.no_normalize:
        normalize = False

    index = faiss.read_index(str(Path(args.index)))

    if index.ntotal != len(documents):
        print(
            f"Warning: index has {index.ntotal} vectors but metadata has {len(documents)} documents"
        )

    model = SentenceTransformer(model_name)
    query_vec = model.encode([args.query], convert_to_numpy=True).astype("float32")

    if normalize:
        faiss.normalize_L2(query_vec)

    needs_filter = bool(args.filter_source or args.filter_term or args.score_threshold is not None)
    fetch_k = args.fetch_k or (args.top_k * 4 if needs_filter else args.top_k)

    scores, indices = index.search(query_vec, fetch_k)

    allowed_sources = {
        s.strip() for s in args.filter_source.split(",") if s.strip()
    }
    allowed_categories = {
        s.strip() for s in args.filter_category.split(",") if s.strip()
    }
    filter_term = args.filter_term.strip().lower()

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        doc = documents[idx]
        source = doc.get("source", "")
        category = doc.get("category", "")
        if allowed_sources and source not in allowed_sources:
            continue
        if allowed_categories and category not in allowed_categories:
            continue
        if filter_term:
            haystack = f"{doc.get('title','')} {doc.get('text','')}".lower()
            if filter_term not in haystack:
                continue
        if args.score_threshold is not None:
            if metric == "cosine" and score < args.score_threshold:
                continue
            if metric == "l2" and score > args.score_threshold:
                continue
        results.append((score, doc))
        if len(results) >= args.top_k:
            break

    print(f"Query: {args.query}")
    if not results:
        print("No results matched the filters.")
        return

    for rank, (score, doc) in enumerate(results, start=1):
        title = doc.get("title", "")
        source = doc.get("source", "")
        category = doc.get("category", "")
        category_label = f" | category={category}" if category else ""
        chunk = doc.get("chunk", None)
        chunk_total = doc.get("chunks_total", None)
        if chunk is not None and chunk_total:
            chunk_label = f" | chunk={chunk}/{chunk_total}"
        elif chunk is not None:
            chunk_label = f" | chunk={chunk}"
        else:
            chunk_label = ""
        print(
            f"[{rank}] score={score:.4f} | {title} | source={source}{category_label}{chunk_label}"
        )
        text = doc.get("text", "")
        if args.show_text:
            print(f"    {clip_text(text, args.max_text)}")


if __name__ == "__main__":
    main()
