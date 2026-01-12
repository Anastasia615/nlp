#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_documents(path, max_docs=None):
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
            if max_docs and len(documents) >= max_docs:
                break
    return documents


def build_text(doc):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title}. {text}"
    return title or text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Index a JSONL document collection into a FAISS vector index."
    )
    parser.add_argument("--data", required=True, help="Path to JSONL documents")
    parser.add_argument("--index", required=True, help="Output path for FAISS index")
    parser.add_argument("--meta", required=True, help="Output path for metadata JSON")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--normalize", action="store_true", help="Force L2 normalization")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization")
    return parser.parse_args()


def main():
    args = parse_args()

    documents = load_documents(args.data, max_docs=args.max_docs or None)
    if not documents:
        raise SystemExit("No documents found. Check the input path.")

    texts = [build_text(doc) for doc in documents]
    if any(not t for t in texts):
        raise SystemExit("One or more documents are missing text or title.")

    normalize = args.metric == "cosine"
    if args.normalize:
        normalize = True
    if args.no_normalize:
        normalize = False

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    if normalize:
        faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    if args.metric == "cosine":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": {
            "model": args.model,
            "metric": args.metric,
            "normalize": normalize,
            "documents": len(documents),
        },
        "documents": documents,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"Indexed {len(documents)} documents")
    print(f"Index saved to {index_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
