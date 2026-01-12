#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a tab-separated news.txt file into JSONL for indexing."
    )
    parser.add_argument("--input", required=True, help="Path to news.txt")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--source", default="news", help="Source label for dataset")
    parser.add_argument("--max-docs", type=int, default=0, help="Limit number of docs")
    parser.add_argument("--skip-empty", action="store_true", help="Skip empty title/text")
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(args.input, "r", encoding="utf-8") as f_in, open(
        out_path, "w", encoding="utf-8"
    ) as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                skipped += 1
                continue
            category, title, text = (p.strip() for p in parts)
            if args.skip_empty and (not title or not text):
                skipped += 1
                continue

            doc = {
                "id": f"news-{line_no:06d}",
                "title": title,
                "text": text,
                "source": args.source,
                "category": category,
            }
            f_out.write(json.dumps(doc, ensure_ascii=True) + "\n")
            written += 1
            if args.max_docs and written >= args.max_docs:
                break

    print(f"Wrote {written} documents to {out_path}")
    if skipped:
        print(f"Skipped {skipped} lines with invalid format")


if __name__ == "__main__":
    main()
