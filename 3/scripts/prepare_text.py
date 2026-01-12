#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
FOOTNOTE_RE = re.compile(r"\[[0-9]+\]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a plain text file into JSONL chunks for indexing."
    )
    parser.add_argument("--input", required=True, help="Path to plain text file")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--title", default="", help="Document title")
    parser.add_argument("--source", default="text", help="Source label for dataset")
    parser.add_argument("--max-chars", type=int, default=800, help="Max chars per chunk")
    return parser.parse_args()


def normalize(text):
    text = FOOTNOTE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_paragraphs(path):
    paragraphs = []
    buffer = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if buffer:
                    paragraphs.append(normalize(" ".join(buffer)))
                    buffer = []
                continue
            buffer.append(line)
    if buffer:
        paragraphs.append(normalize(" ".join(buffer)))
    return [p for p in paragraphs if p]


def split_long_paragraph(paragraph, max_chars):
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = SENTENCE_SPLIT_RE.split(paragraph)
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) <= max_chars:
                current = sentence
            else:
                for i in range(0, len(sentence), max_chars):
                    part = sentence[i : i + max_chars].strip()
                    if part:
                        chunks.append(part)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def build_chunks(paragraphs, max_chars):
    chunks = []
    current = ""
    for paragraph in paragraphs:
        parts = split_long_paragraph(paragraph, max_chars)
        for part in parts:
            if len(current) + len(part) + 1 <= max_chars:
                current = f"{current} {part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part
    if current:
        chunks.append(current)
    return chunks


def main():
    args = parse_args()
    paragraphs = read_paragraphs(args.input)
    if not paragraphs:
        raise SystemExit("No text found in input file.")

    chunks = build_chunks(paragraphs, args.max_chars)
    if not chunks:
        raise SystemExit("No chunks were created from input file.")

    base_title = args.title.strip() or Path(args.input).stem
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            doc = {
                "id": f"{base_title.lower()}-{idx:04d}",
                "title": f"{base_title} (chunk {idx}/{total})",
                "text": chunk,
                "source": args.source,
                "chunk": idx,
                "chunks_total": total,
            }
            f_out.write(json.dumps(doc, ensure_ascii=True) + "\n")

    print(f"Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
