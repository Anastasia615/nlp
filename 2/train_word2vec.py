#!/usr/bin/env python3
import argparse
import gzip
import os
from pathlib import Path
import re
import sys

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token for token in WORD_RE.findall(text.lower()) if token.isalpha()]


class NewsCorpus:
    def __init__(self, path: Path, encoding: str) -> None:
        self.path = path
        self.encoding = encoding

    def __iter__(self):
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding=self.encoding, errors="ignore") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    text = f"{parts[1]} {parts[2]}"
                elif len(parts) == 2:
                    text = f"{parts[0]} {parts[1]}"
                else:
                    text = parts[0]
                tokens = tokenize(text)
                if tokens:
                    yield tokens


def build_default_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    default_corpus = base_dir / "nlp-2025" / "data" / "news.txt.gz"
    if not default_corpus.exists():
        alt = base_dir / "data" / "news.txt.gz"
        if alt.exists():
            default_corpus = alt
    default_model = base_dir / "models" / "word2vec.model"
    return default_corpus, default_model


def parse_args() -> argparse.Namespace:
    default_corpus, default_model = build_default_paths()
    parser = argparse.ArgumentParser(
        description="Train a Word2Vec model on a news corpus."
    )
    parser.add_argument(
        "--corpus",
        default=str(default_corpus),
        help="Path to corpus (.txt or .gz).",
    )
    parser.add_argument(
        "--model",
        default=str(default_model),
        help="Where to save the trained model.",
    )
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--sg",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 = skip-gram, 0 = CBOW",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoding", default="utf-8")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus file not found: {corpus_path}", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from gensim.models import Word2Vec
    except Exception as exc:
        print(f"Failed to import gensim: {exc}", file=sys.stderr)
        return 1

    sentences = NewsCorpus(corpus_path, args.encoding)
    try:
        model = Word2Vec(
            sentences=sentences,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            sg=args.sg,
            epochs=args.epochs,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    try:
        model.save(str(model_path))
    except Exception as exc:
        print(f"Failed to save model to {model_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Model saved to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
