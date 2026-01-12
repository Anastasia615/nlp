#!/usr/bin/env python3
import argparse
from functools import lru_cache
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


def load_lexicon(use_lexicon: bool):
    if not use_lexicon:
        return None
    try:
        from ruwordnet import RuWordNet
    except Exception as exc:
        print(f"Lexicon disabled: failed to import ruwordnet: {exc}", file=sys.stderr)
        return None
    try:
        return RuWordNet()
    except FileNotFoundError:
        print(
            "Lexicon disabled: RuWordNet database not found. Run: ruwordnet download",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        print(f"Lexicon disabled: failed to initialize RuWordNet: {exc}", file=sys.stderr)
        return None


def load_morph(use_morph: bool):
    if not use_morph:
        return None
    try:
        import pymorphy3
    except Exception as exc:
        print(f"Lemmatization disabled: failed to import pymorphy3: {exc}", file=sys.stderr)
        return None
    try:
        return pymorphy3.MorphAnalyzer()
    except Exception as exc:
        print(f"Lemmatization disabled: failed to initialize pymorphy3: {exc}", file=sys.stderr)
        return None


def make_lemmatizer(morph):
    if morph is None:
        return lambda token: token

    @lru_cache(maxsize=200000)
    def lemmatize(token: str) -> str:
        if not token:
            return token
        if script_type(token) != "cyrillic":
            return token
        try:
            return morph.parse(token)[0].normal_form
        except Exception:
            return token

    return lemmatize


def lexicon_synonyms(tokens: set[str], lexicon, allow_cross_script: bool, max_per_token: int) -> set[str]:
    if lexicon is None:
        return set()
    synonyms = set()
    for token in tokens:
        token_script = script_type(token)
        added = 0
        for synset in lexicon.get_synsets(token):
            for sense in synset.senses:
                candidate = sense.lemma.lower()
                parts = tokenize(candidate)
                if len(parts) != 1:
                    continue
                candidate = parts[0]
                if not allow_cross_script and token_script in {"cyrillic", "latin"}:
                    if script_type(candidate) != token_script:
                        continue
                if candidate != token:
                    synonyms.add(candidate)
                    added += 1
                    if max_per_token and added >= max_per_token:
                        break
            if max_per_token and added >= max_per_token:
                break
    return synonyms


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def iter_alpha_tokens(text: str):
    for match in WORD_RE.finditer(text):
        token = match.group(0)
        if token.isalpha():
            yield token


def sentence_tokens(text: str, lemmatize=None):
    pairs = []
    for token in iter_alpha_tokens(text):
        normalized = token.lower()
        lemma = lemmatize(normalized) if lemmatize else normalized
        pairs.append((token, normalized, lemma))
    return pairs


def cosine_similarity(vec1, vec2) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def vector_key(model, token: str, lemma: str):
    if model is None:
        return None
    if token in model.wv:
        return token
    if lemma and lemma in model.wv:
        return lemma
    return None


def context_vector(model, tokens: list[str], lemmas: list[str], index: int, window: int, min_tokens: int):
    if model is None or window <= 0:
        return None
    start = max(0, index - window)
    end = min(len(tokens), index + window + 1)
    vecs = []
    for idx in range(start, end):
        if idx == index:
            continue
        token = tokens[idx]
        lemma = lemmas[idx] if lemmas else token
        key = vector_key(model, token, lemma)
        if key:
            vecs.append(model.wv[key])
    if len(vecs) < min_tokens:
        return None
    return np.mean(vecs, axis=0)


def similarity_to_context(model, word: str, lemma: str, context_vec):
    if model is None or context_vec is None:
        return None
    key = vector_key(model, word, lemma)
    if key is None:
        return None
    return cosine_similarity(model.wv[key], context_vec)


def find_match_token(
    text: str,
    query_terms: set[str],
    synonym_forms: set[str],
    synonym_lemmas: set[str],
    exact_lemmas: set[str],
    synonym_sources,
    model,
    context_window: int,
    context_threshold: float,
    context_source_threshold: float,
    context_delta: float,
    context_min_tokens: int,
    lemmatize,
):
    token_pairs = sentence_tokens(text, lemmatize)
    if not token_pairs:
        return None
    normalized_tokens = [normalized for _, normalized, _ in token_pairs]
    lemmas = [lemma for _, _, lemma in token_pairs]
    first_exact = None
    first_syn = None
    for idx, (raw_token, normalized, lemma) in enumerate(token_pairs):
        is_exact = normalized in query_terms or (exact_lemmas and lemma in exact_lemmas)
        if is_exact:
            if first_exact is None:
                first_exact = raw_token
            continue
        is_syn = normalized in synonym_forms or (synonym_lemmas and lemma in synonym_lemmas)
        if not is_syn:
            continue
        context_vec = context_vector(
            model,
            normalized_tokens,
            lemmas,
            idx,
            context_window,
            context_min_tokens,
        )
        if context_vec is None:
            if first_syn is None:
                first_syn = raw_token
            continue
        score = similarity_to_context(model, normalized, lemma, context_vec)
        if score is None or score < context_threshold:
            continue
        key = lemma if synonym_lemmas and lemma in synonym_lemmas else normalized
        sources = synonym_sources.get(key, set())
        if sources:
            best_source = None
            for source in sources:
                source_lemma = lemmatize(source) if lemmatize else source
                source_score = similarity_to_context(model, source, source_lemma, context_vec)
                if source_score is None:
                    continue
                if best_source is None or source_score > best_source:
                    best_source = source_score
            if best_source is None or best_source < context_source_threshold:
                continue
            if score - best_source > context_delta:
                continue
        if first_syn is None:
            first_syn = raw_token
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
        "--lexicon",
        action="store_true",
        dest="use_lexicon",
        help="Enable RuWordNet-based synonyms.",
    )
    parser.add_argument(
        "--lexicon-max",
        type=int,
        default=20,
        help="Maximum number of lexicon synonyms per query token.",
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
        dest="sentence",
        help="Print only sentence(s) containing the query terms.",
    )
    parser.add_argument(
        "--full-line",
        action="store_false",
        dest="sentence",
        help="Print full matching lines instead of sentences.",
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
    parser.set_defaults(sentence=True)
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
    query_terms: set[str],
    synonym_forms: set[str],
    synonym_lemmas: set[str],
    exact_lemmas: set[str],
    synonym_sources,
    model,
    context_window: int,
    context_threshold: float,
    context_source_threshold: float,
    context_delta: float,
    context_min_tokens: int,
    lemmatize,
):
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return []
    matches = []
    for sentence in sentences:
        match = find_match_token(
            sentence,
            query_terms,
            synonym_forms,
            synonym_lemmas,
            exact_lemmas,
            synonym_sources,
            model,
            context_window,
            context_threshold,
            context_source_threshold,
            context_delta,
            context_min_tokens,
            lemmatize,
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
    if args.lexicon_max < 0:
        print("Lexicon max must be >= 0.", file=sys.stderr)
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

    wn = load_lexicon(args.use_lexicon)
    lexicon_active = args.use_lexicon and wn is not None
    morph = load_morph(args.use_lexicon)
    lemmatize = make_lemmatizer(morph)
    lemma_enabled = lexicon_active and morph is not None
    query_terms = set(tokens)
    query_lemmas = {lemmatize(token) for token in tokens} if lemma_enabled else set()
    lexicon_terms = set()
    if lexicon_active:
        lexicon_seed = query_lemmas if lemma_enabled else query_terms
        lexicon_terms = set(
            lexicon_synonyms(lexicon_seed, wn, args.cross_script, args.lexicon_max)
        )

    model = None
    synonym_sources = {}
    w2v_candidates = set()
    if not args.exact:
        seed_tokens = list(tokens)
        try:
            model = load_model(Path(args.model))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _, oov, rare, synonym_pairs = expand_terms(
            model,
            seed_tokens,
            args.topn,
            args.threshold,
            args.cross_script,
            args.min_freq,
            args.mutual_topn,
        )
        normalized_pairs = []
        for source, syn in synonym_pairs:
            if source and syn and source != syn:
                normalized_pairs.append((source, syn))
        synonym_sources = {}
        for source_norm, syn_norm in normalized_pairs:
            w2v_candidates.add(syn_norm)
            key = lemmatize(syn_norm) if lemma_enabled else syn_norm
            synonym_sources.setdefault(key, set()).add(source_norm)
    else:
        rare = []
        oov = []

    synonym_forms = set()
    synonym_lemmas = set()
    if lexicon_active:
        if lemma_enabled:
            candidate_lemmas = {lemmatize(word) for word in w2v_candidates}
            synonym_lemmas = (candidate_lemmas & lexicon_terms) - query_lemmas
            synonym_forms = {word for word in w2v_candidates if lemmatize(word) in synonym_lemmas}
            synonym_sources = {
                key: value for key, value in synonym_sources.items() if key in synonym_lemmas
            }
        else:
            synonym_forms = w2v_candidates & lexicon_terms
            synonym_sources = {
                key: value for key, value in synonym_sources.items() if key in synonym_forms
            }
    else:
        synonym_forms = set(w2v_candidates)

    synonyms = synonym_lemmas if lemma_enabled else synonym_forms
    exact_lemmas = query_lemmas if lemma_enabled else set()
    terms_forms = set(query_terms) | synonym_forms
    terms_lemmas = (exact_lemmas | synonym_lemmas) if lemma_enabled else set()

    if args.show_terms:
        print(f"Query tokens: {', '.join(tokens)}", file=sys.stderr)
        if w2v_candidates:
            print(
                f"Word2Vec candidates ({len(w2v_candidates)}): {', '.join(sorted(w2v_candidates))}",
                file=sys.stderr,
            )
        else:
            print("Word2Vec candidates: none", file=sys.stderr)
        if lemma_enabled:
            if query_lemmas:
                print(f"Query lemmas: {', '.join(sorted(query_lemmas))}", file=sys.stderr)
            if synonym_forms:
                print(
                    f"Synonym forms ({len(synonym_forms)}): {', '.join(sorted(synonym_forms))}",
                    file=sys.stderr,
                )
        if lexicon_active:
            if lexicon_terms:
                print(
                    f"Lexicon synonyms ({len(lexicon_terms)}): {', '.join(sorted(lexicon_terms))}",
                    file=sys.stderr,
                )
            else:
                print("Lexicon synonyms: none", file=sys.stderr)
            if synonyms:
                print(
                    f"Filtered synonyms ({len(synonyms)}): {', '.join(sorted(synonyms))}",
                    file=sys.stderr,
                )
            else:
                print("Filtered synonyms: none", file=sys.stderr)
        else:
            if synonyms:
                print(f"Synonyms ({len(synonyms)}): {', '.join(sorted(synonyms))}", file=sys.stderr)
            else:
                print("Synonyms: none", file=sys.stderr)
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
                if lemma_enabled:
                    tokens_in_line = [token.lower() for token in iter_alpha_tokens(text)]
                    if not tokens_in_line:
                        continue
                    forms_hit = set(tokens_in_line) & terms_forms
                    lemmas_hit = {lemmatize(token) for token in tokens_in_line} & terms_lemmas
                    if not (forms_hit or lemmas_hit):
                        continue
                else:
                    if not (token_set(text) & terms_forms):
                        continue
                if args.sentence:
                    matches = sentences_with_terms(
                        text,
                        query_terms,
                        synonym_forms,
                        synonym_lemmas,
                        exact_lemmas,
                        synonym_sources,
                        model,
                        args.context_window,
                        args.context_threshold,
                        args.context_source_threshold,
                        args.context_delta,
                        args.context_min_tokens,
                        lemmatize,
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
