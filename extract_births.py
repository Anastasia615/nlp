#!/usr/bin/env python3
import argparse
import json
import re

from yargy import Parser, or_, rule
from yargy.interpretation import fact
from yargy.predicates import caseless, custom, dictionary, eq, gram, is_capitalized, normalized


PRONOUN_RE = re.compile(r"^\s*(он|она)\b", re.IGNORECASE)
KEYWORDS_GROUP = (
    r"(родился\w*|родилася\w*|родивш\w*|урожен\w*|родом|род\."
    r"|(?:\(|\s)р\.(?=\s*\d)"
    r"|выходец\w*|на свет)"
)
KEYWORD_RE = re.compile(KEYWORDS_GROUP, re.IGNORECASE)
LATIN_PARTICLES = {"van", "von", "de", "da", "del", "la", "le", "di", "du"}


def build_parsers():
    Entry = fact("Entry", ["name", "birth_date", "birth_place"])
    NameOnly = fact("NameOnly", ["name"])

    def is_hyphen(value):
        return value in {"-", "‑"}

    hyphen = custom(is_hyphen)
    comma = eq(",")
    dash = eq("—")
    dot = eq(".")
    lparen = eq("(")
    rparen = eq(")")

    def is_apostrophe(value):
        return value in {"'", "’"}

    apostrophe = custom(is_apostrophe)

    def is_title(value):
        return value[:1].isupper() and not value.isupper()

    def is_initial(value):
        return len(value) == 1 and value.isalpha() and value.isupper()

    latin_title = custom(is_title, types="LATIN")
    initial_letter = custom(is_initial, types=["RU", "LATIN"])
    initial = rule(initial_letter, dot)

    name_atom = or_(
        rule(gram("Name")),
        rule(gram("Surn")),
        rule(gram("Patr")),
        rule(latin_title),
    )
    name_word = or_(
        name_atom,
        rule(name_atom, hyphen, name_atom),
        rule(name_atom, apostrophe, name_atom),
    )
    particle = or_(
        rule(caseless("van")),
        rule(caseless("von")),
        rule(caseless("de")),
        rule(caseless("da")),
        rule(caseless("del")),
        rule(caseless("la")),
        rule(caseless("le")),
        rule(caseless("di")),
        rule(caseless("du")),
    )
    name_head = name_atom
    name_full = rule(
        name_head,
        particle.optional(),
        name_word,
        name_word.optional(),
    )
    name_initials = rule(
        initial,
        initial.optional(),
        name_word,
    )
    name_base = or_(name_full, name_initials)
    name_entry = name_base.interpretation(Entry.name)
    name_only = name_base.interpretation(NameOnly.name)

    def is_year(value):
        return value.isdigit() and len(value) == 4

    def is_day(value):
        if not value.isdigit():
            return False
        value = int(value)
        return 1 <= value <= 31

    year = custom(is_year, types="INT")
    day = custom(is_day, types="INT")
    month_gen = dictionary(
        [
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ]
    )
    month_prep = dictionary(
        [
            "январе",
            "феврале",
            "марте",
            "апреле",
            "мае",
            "июне",
            "июле",
            "августе",
            "сентябре",
            "октябре",
            "ноябре",
            "декабре",
        ]
    )
    year_word = dictionary(["год", "года", "г.", "году", "годах", "годов"])
    year_suffix = dictionary(["м", "й", "х", "е"])

    date_dmy = rule(day, month_gen, year, year_word.optional())
    date_my = rule(month_prep, year, year_word.optional())
    date_y = rule(year, year_word.optional())
    date_short = rule(year, hyphen, year_suffix, year_word.optional())
    date = or_(date_dmy, date_my, date_short, date_y).interpretation(Entry.birth_date)

    place_descriptor = dictionary(
        [
            "город",
            "городе",
            "село",
            "селе",
            "деревня",
            "деревне",
            "поселок",
            "посёлок",
            "поселке",
            "посёлке",
            "штат",
            "штате",
            "область",
            "области",
            "край",
            "крае",
            "республика",
            "республике",
            "округ",
            "округе",
            "район",
            "районе",
            "провинция",
            "провинции",
            "остров",
            "острове",
            "столица",
            "столице",
            "губерния",
            "губернии",
        ]
    )
    place_part = is_capitalized()
    place_abbr = or_(
        rule(caseless("г"), dot),
        rule(caseless("пос"), dot),
        rule(caseless("р"), hyphen, caseless("н"), dot.optional()),
        rule(caseless("обл"), dot),
    )
    place_word = or_(
        rule(place_part),
        rule(place_part, hyphen, place_part),
        rule(place_descriptor),
        place_abbr,
    )
    place = rule(
        place_word,
        place_word.optional(),
        place_word.optional(),
        place_word.optional(),
        place_word.optional(),
    ).interpretation(Entry.birth_place)

    prep_in = or_(caseless("в"), caseless("во"), caseless("на"))
    prep_from = caseless("из")
    born_word = or_(
        rule(normalized("родиться")),
        rule(normalized("появиться"), caseless("на"), caseless("свет")),
        rule(normalized("прийти"), caseless("на"), caseless("свет")),
    )
    born_abbr = or_(
        rule(caseless("род"), dot),
        rule(caseless("р"), dot),
    )
    native = or_(normalized("уроженец"), normalized("уроженка"))
    exit_from = or_(normalized("выходец"), caseless("выходцы"))
    from_place = normalized("родом")

    opt_punct = or_(rule(comma), rule(dash)).optional()
    opt_sep = or_(rule(comma), rule(dash)).optional()

    entry_patterns = [
        rule(name_entry, opt_punct, born_word, prep_in, place, opt_sep, date.optional()),
        rule(name_entry, opt_punct, born_word, prep_in, date, prep_in, place),
        rule(name_entry, opt_punct, born_word, prep_in, date),
        rule(name_entry, opt_punct, born_word, date, prep_in, place),
        rule(name_entry, opt_punct, born_word, date),
        rule(name_entry, opt_punct, native, place),
        rule(name_entry, opt_punct, exit_from, prep_from, place),
        rule(name_entry, opt_punct, from_place, prep_from, place),
        rule(name_entry, opt_punct, born_abbr, date, opt_sep, prep_in.optional(), place.optional()),
        rule(name_entry, opt_punct, lparen, born_abbr, date, opt_sep, prep_in.optional(), place.optional(), rparen),
    ]

    pronoun = or_(rule(caseless("он")), rule(caseless("она")))
    pronoun_patterns = [
        rule(pronoun, opt_punct, born_word, prep_in, place, opt_sep, date.optional()),
        rule(pronoun, opt_punct, born_word, prep_in, date, prep_in, place),
        rule(pronoun, opt_punct, born_word, prep_in, date),
        rule(pronoun, opt_punct, born_word, date, prep_in, place),
        rule(pronoun, opt_punct, born_word, date),
        rule(pronoun, opt_punct, native, place),
        rule(pronoun, opt_punct, exit_from, prep_from, place),
        rule(pronoun, opt_punct, from_place, prep_from, place),
    ]

    entry_parser = Parser(or_(*entry_patterns).interpretation(Entry))
    name_parser = Parser(name_only)
    pronoun_parser = Parser(or_(*pronoun_patterns).interpretation(Entry))
    return entry_parser, name_parser, pronoun_parser


def parse_args():
    parser = argparse.ArgumentParser(description="Extract birth info from news.txt using Yargy.")
    parser.add_argument("--input", default="news.txt", help="Path to news.txt")
    parser.add_argument("--output", default="entries.json", help="Output JSON file")
    return parser.parse_args()


def normalize_value(value):
    if value is None:
        return None
    return value.strip()


def is_initial_token(token):
    return len(token) == 2 and token[1] == "." and token[0].isalpha() and token[0].isupper()


def is_title_token(token):
    if is_initial_token(token):
        return True
    lowered = token.lower()
    if lowered in LATIN_PARTICLES:
        return True
    if "-" in token or "‑" in token:
        parts = re.split(r"[-‑]", token)
        return all(is_title_token(part) for part in parts if part)
    if "'" in token or "’" in token:
        normalized = token.replace("’", "'")
        return all(part.istitle() for part in normalized.split("'") if part)
    return token.istitle()


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def keyword_windows(text, window=120):
    spans = []
    for match in KEYWORD_RE.finditer(text):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        spans.append((start, end))
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [text[start:end] for start, end in merged]


def extract_entries(parsers, text):
    entry_parser, name_parser, pronoun_parser = parsers
    entries = []
    for snippet in keyword_windows(text):
        for match in entry_parser.findall(snippet):
            entry = match.fact
            name = normalize_value(getattr(entry, "name", None))
            birth_date = normalize_value(getattr(entry, "birth_date", None))
            birth_place = normalize_value(getattr(entry, "birth_place", None))
            if (
                not name
                or any(not is_title_token(token) for token in name.split())
                or (birth_date is None and birth_place is None)
            ):
                continue
            entries.append(
                {
                    "name": name,
                    "birth_date": birth_date,
                    "birth_place": birth_place,
                }
            )

    last_name = None
    for sentence in split_sentences(text):
        for match in name_parser.findall(sentence):
            candidate = normalize_value(match.fact)
            if not candidate or any(not is_title_token(token) for token in candidate.split()):
                continue
            last_name = candidate
        if not last_name or not PRONOUN_RE.match(sentence):
            continue
        for match in pronoun_parser.findall(sentence):
            entry = match.fact
            birth_date = normalize_value(getattr(entry, "birth_date", None))
            birth_place = normalize_value(getattr(entry, "birth_place", None))
            if birth_date is None and birth_place is None:
                continue
            entries.append(
                {
                    "name": last_name,
                    "birth_date": birth_date,
                    "birth_place": birth_place,
                }
            )

    return entries


def main():
    args = parse_args()
    parsers = build_parsers()

    results = []
    seen = set()
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            _, title, body = parts
            text = f"{title} {body}"
            if not KEYWORD_RE.search(text):
                continue
            for entry in extract_entries(parsers, text):
                key = (entry["name"], entry["birth_date"], entry["birth_place"])
                if key in seen:
                    continue
                seen.add(key)
                results.append(entry)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
