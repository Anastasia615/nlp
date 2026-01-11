#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path
import re
import sys

import numpy as np

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])(?=[A-ZА-ЯЁ])")
CYRILLIC_RE = re.compile(r"[А-ЯЁа-яё]")
LATIN_RE = re.compile(r"[A-Za-z]")


def tokenize(text: str) -> list[str]:
    return [token for token in WORD_RE.findall(text.lower()) if token.isalpha()]


def open_text(path: Path, encoding: str):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding=encoding, errors="strict")


def load_model(path: Path):
    try:
        from gensim.models import Word2Vec
    except Exception as exc:
        raise RuntimeError(f"Failed to import gensim: {exc}") from exc

    try:
        return Word2Vec.load(str(path))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Model file not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


def script_type(token: str) -> str:
    has_cyrillic = bool(CYRILLIC_RE.search(token))
    has_latin = bool(LATIN_RE.search(token))
    if has_cyrillic and not has_latin:
        return "cyrillic"
    if has_latin and not has_cyrillic:
        return "latin"
    if has_cyrillic or has_latin:
        return "mixed"
    return "other"


def load_morph(use_morph: bool):
    if not use_morph:
        return None
    last_exc = None
    for lib_name in ("pymorphy3", "pymorphy2"):
        try:
            module = __import__(lib_name)
        except Exception as exc:
            last_exc = exc
            continue
        try:
            return module.MorphAnalyzer()
        except Exception as exc:
            last_exc = exc
            continue
    print(
        f"Lemmatization disabled: failed to initialize pymorphy2/pymorphy3: {last_exc}",
        file=sys.stderr,
    )
    return None


def lemmatize(token: str, morph) -> str:
    if morph is None:
        return token
    try:
        parses = morph.parse(token)
    except Exception:
        return token
    if not parses:
        return token
    return parses[0].normal_form


def normalize_tokens(tokens: list[str], morph, use_lemma: bool) -> list[str]:
    if not use_lemma or morph is None:
        return tokens
    return [lemmatize(token, morph) for token in tokens]


def token_set(text: str, morph, use_lemma: bool) -> set[str]:
    return set(normalize_tokens(tokenize(text), morph, use_lemma))


def iter_alpha_tokens(text: str):
    for match in WORD_RE.finditer(text):
        token = match.group(0)
        if token.isalpha():
            yield token


def sentence_tokens(text: str, morph, use_lemma: bool):
    items = []
    for token in iter_alpha_tokens(text):
        lower = token.lower()
        lemma = lemmatize(lower, morph) if use_lemma and morph is not None else lower
        items.append((token, lemma))
    return items


def cosine_similarity(vec1, vec2) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def context_vector(model, lemmas: list[str], index: int, window: int, min_tokens: int):
    if model is None or window <= 0:
        return None
    start = max(0, index - window)
    end = min(len(lemmas), index + window + 1)
    vecs = []
    for idx in range(start, end):
        if idx == index:
            continue
        lemma = lemmas[idx]
        if lemma in model.wv:
            vecs.append(model.wv[lemma])
    if len(vecs) < min_tokens:
        return None
    return np.mean(vecs, axis=0)


def similarity_to_context(model, word: str, context_vec):
    if model is None or context_vec is None:
        return None
    if word not in model.wv:
        return None
    return cosine_similarity(model.wv[word], context_vec)


def find_match_token(
    text: str,
    terms: set[str],
    synonyms: set[str],
    synonym_sources,
    morph,
    use_lemma: bool,
    model,
    context_window: int,
    context_threshold: float,
    context_source_threshold: float,
    context_delta: float,
    context_min_tokens: int,
):
    items = sentence_tokens(text, morph, use_lemma)
    if not items:
        return None
    lemmas = [lemma for _, lemma in items]
    first_exact = None
    first_syn = None
    for idx, (token, lemma) in enumerate(items):
        if lemma not in terms:
            continue
        if lemma in synonyms:
            context_vec = context_vector(
                model,
                lemmas,
                idx,
                context_window,
                context_min_tokens,
            )
            if context_vec is None:
                if first_syn is None:
                    first_syn = token
                continue
            score = similarity_to_context(model, lemma, context_vec)
            if score is None or score < context_threshold:
                continue
            sources = synonym_sources.get(lemma, set())
            if sources:
                best_source = None
                for source in sources:
                    source_score = similarity_to_context(model, source, context_vec)
                    if source_score is None:
                        continue
                    if best_source is None or source_score > best_source:
                        best_source = source_score
                if best_source is None or best_source < context_source_threshold:
                    continue
                if score - best_source > context_delta:
                    continue
            if first_syn is None:
                first_syn = token
        else:
            if first_exact is None:
                first_exact = token
    if first_syn is not None:
        return first_syn, True
    if first_exact is not None:
        return first_exact, False
    return None


def expand_terms(
    model,
    tokens: list[str],
    topn: int,
    threshold: float,
    allow_cross_script: bool,
    min_freq: int,
    mutual_topn: int,
):
    terms = set(tokens)
    oov = []
    rare = []
    synonym_pairs = []
    mutual_cache = {}
    for token in tokens:
        token_script = script_type(token)
        if token not in model.wv:
            oov.append(token)
            continue
        count = model.wv.get_vecattr(token, "count")
        if count < min_freq:
            rare.append((token, count))
            continue
        try:
            similar = model.wv.most_similar(token, topn=topn)
        except KeyError:
            oov.append(token)
            continue
        for word, score in similar:
            if not allow_cross_script and token_script in {"cyrillic", "latin"}:
                if script_type(word) != token_script:
                    continue
            if mutual_topn > 0:
                neighbors = mutual_cache.get(word)
                if neighbors is None:
                    try:
                        neighbors = {w for w, _ in model.wv.most_similar(word, topn=mutual_topn)}
                    except KeyError:
                        neighbors = set()
                    mutual_cache[word] = neighbors
                if token not in neighbors:
                    continue
            if score >= threshold:
                terms.add(word)
                synonym_pairs.append((token, word))
    return terms, oov, rare, synonym_pairs


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_model = base_dir / "models" / "word2vec.model"
    parser = argparse.ArgumentParser(
        description="Search lines in a text file using Word2Vec synonyms."
    )
    parser.add_argument("file", help="Path to a text file (.txt or .gz).")
    parser.add_argument("query", help="Query word or words.")
    parser.add_argument("--model", default=str(default_model))
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--show-terms",
        action="store_true",
        help="Print expanded terms to stderr.",
    )
    parser.add_argument(
        "--cross-script",
        action="store_true",
        help="Allow expanded terms in other scripts (e.g., Latin for Cyrillic query).",
    )
    parser.add_argument(
        "--no-lemma",
        action="store_false",
        dest="use_lemma",
        help="Disable lemmatization for matching.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Context window size for synonym validation (0 = disable).",
    )
    parser.add_argument(
        "--context-threshold",
        type=float,
        default=0.3,
        help="Minimum cosine similarity of synonym to context.",
    )
    parser.add_argument(
        "--context-source-threshold",
        type=float,
        default=0.2,
        help="Minimum cosine similarity of query term to context.",
    )
    parser.add_argument(
        "--context-delta",
        type=float,
        default=0.3,
        help="Maximum allowed gap between synonym and query similarity to context.",
    )
    parser.add_argument(
        "--context-min-tokens",
        type=int,
        default=2,
        help="Minimum context tokens with vectors to apply validation.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=10,
        help="Minimum token frequency to allow synonym expansion.",
    )
    parser.add_argument(
        "--mutual-topn",
        type=int,
        default=20,
        help="Keep only mutually similar words (0 = disable).",
    )
    parser.add_argument(
        "--fields",
        choices=["all", "category", "headline", "text", "category_headline"],
        default="all",
        help="Which parts to print for tab-separated news lines.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="Truncate output to this length (0 = no limit).",
    )
    parser.add_argument(
        "--sentence",
        action="store_true",
        help="Print only sentence(s) containing the query terms.",
    )
    parser.add_argument(
        "--only-synonyms",
        action="store_true",
        help="Print only expanded synonyms and exit.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Search only exact tokens without loading a model.",
    )
    return parser.parse_args()


def format_line(line: str, fields: str) -> str:
    raw = line.rstrip("\n")
    if fields == "all":
        return raw
    parts = raw.split("\t")
    category = parts[0] if len(parts) > 0 else ""
    headline = parts[1] if len(parts) > 1 else ""
    text = parts[2] if len(parts) > 2 else ""
    if fields == "category":
        return category
    if fields == "headline":
        return headline or raw
    if fields == "text":
        return text or raw
    if fields == "category_headline":
        if category and headline:
            return f"{category}\t{headline}"
        return raw
    return raw


def truncate(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3].rstrip() + "..."


def sentences_with_terms(
    text: str,
    terms: set[str],
    synonyms: set[str],
    synonym_sources,
    morph,
    use_lemma: bool,
    model,
    context_window: int,
    context_threshold: float,
    context_source_threshold: float,
    context_delta: float,
    context_min_tokens: int,
):
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return []
    matches = []
    for sentence in sentences:
        match = find_match_token(
            sentence,
            terms,
            synonyms,
            synonym_sources,
            morph,
            use_lemma,
            model,
            context_window,
            context_threshold,
            context_source_threshold,
            context_delta,
            context_min_tokens,
        )
        if match:
            matches.append((sentence, match[0], match[1]))
    return matches


def main() -> int:
    args = parse_args()
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Input file not found: {file_path}", file=sys.stderr)
        return 1

    if not (0.0 <= args.threshold <= 1.0):
        print("Threshold must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.context_window < 0:
        print("Context window must be >= 0.", file=sys.stderr)
        return 1
    if not (0.0 <= args.context_threshold <= 1.0):
        print("Context threshold must be between 0 and 1.", file=sys.stderr)
        return 1
    if not (0.0 <= args.context_source_threshold <= 1.0):
        print("Context source threshold must be between 0 and 1.", file=sys.stderr)
        return 1
    if not (0.0 <= args.context_delta <= 1.0):
        print("Context delta must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.context_min_tokens < 0:
        print("Context min tokens must be >= 0.", file=sys.stderr)
        return 1
    if args.mutual_topn < 0:
        print("Mutual topn must be >= 0.", file=sys.stderr)
        return 1

    tokens = tokenize(args.query)
    if not tokens:
        print("Query has no searchable tokens.", file=sys.stderr)
        return 1

    if args.only_synonyms and args.exact:
        print("Cannot use --only-synonyms with --exact.", file=sys.stderr)
        return 1

    morph = load_morph(args.use_lemma)
    query_terms = set(normalize_tokens(tokens, morph, args.use_lemma))

    model = None
    synonym_sources = {}
    if not args.exact:
        seed_tokens = list(tokens)
        if args.use_lemma and morph is not None:
            for lemma in normalize_tokens(tokens, morph, True):
                if lemma not in seed_tokens:
                    seed_tokens.append(lemma)
        try:
            model = load_model(Path(args.model))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        terms, oov, rare, synonym_pairs = expand_terms(
            model,
            seed_tokens,
            args.topn,
            args.threshold,
            args.cross_script,
            args.min_freq,
            args.mutual_topn,
        )
        terms = set(normalize_tokens(list(terms), morph, args.use_lemma))
        normalized_pairs = []
        for source, syn in synonym_pairs:
            source_norm = lemmatize(source, morph) if args.use_lemma and morph is not None else source
            syn_norm = lemmatize(syn, morph) if args.use_lemma and morph is not None else syn
            if source_norm and syn_norm and source_norm != syn_norm:
                normalized_pairs.append((source_norm, syn_norm))
        synonym_sources = {}
        for source_norm, syn_norm in normalized_pairs:
            synonym_sources.setdefault(syn_norm, set()).add(source_norm)
    else:
        terms = set(query_terms)
        rare = []
        oov = []

    synonyms = set(synonym_sources.keys())

    if args.show_terms:
        expanded = sorted(terms - query_terms)
        if args.use_lemma and morph is not None:
            print(f"Query tokens: {', '.join(tokens)}", file=sys.stderr)
            print(f"Query lemmas: {', '.join(sorted(query_terms))}", file=sys.stderr)
        else:
            print(f"Query tokens: {', '.join(tokens)}", file=sys.stderr)
        if expanded:
            print(f"Expanded terms ({len(expanded)}): {', '.join(expanded)}", file=sys.stderr)
        else:
            print("Expanded terms: none", file=sys.stderr)
        if oov:
            print(f"OOV tokens: {', '.join(oov)}", file=sys.stderr)
        if rare:
            desc = ", ".join(f"{token}({count})" for token, count in rare)
            print(f"Rare tokens (no expansion): {desc}", file=sys.stderr)

    if args.only_synonyms:
        if synonyms:
            for word in sorted(synonyms):
                print(word)
        else:
            print("No synonyms found.", file=sys.stderr)
        return 0

    try:
        with open_text(file_path, args.encoding) as handle:
            index = 0
            effective_fields = args.fields
            if args.sentence and args.fields == "all":
                effective_fields = "text"
            for line in handle:
                text = format_line(line, effective_fields)
                if not (token_set(text, morph, args.use_lemma) & terms):
                    continue
                if args.sentence:
                    matches = sentences_with_terms(
                        text,
                        terms,
                        synonyms,
                        synonym_sources,
                        morph,
                        args.use_lemma,
                        model,
                        args.context_window,
                        args.context_threshold,
                        args.context_source_threshold,
                        args.context_delta,
                        args.context_min_tokens,
                    )
                    for sentence, match_token, is_syn in matches:
                        sentence = truncate(sentence, args.max_len)
                        label = "синоним" if is_syn else "совпадение"
                        sentence = f"{sentence} ({label}: {match_token})"
                        index += 1
                        print(f"{index}. {sentence}")
                else:
                    text = truncate(text, args.max_len)
                    index += 1
                    print(f"{index}. {text}")
    except UnicodeDecodeError as exc:
        print(
            f"Failed to decode file with encoding {args.encoding}: {exc}",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"Failed to read file {file_path}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
