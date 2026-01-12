# Семантический поиск (векторный)

Минимальная система семантического поиска:
- эмбеддинги предложений (SentenceTransformers)
- векторный индекс (FAISS)
- отдельные CLI-скрипты для индексации и поиска

В проекте есть демонстрационный набор (`data/documents.jsonl`) и поддержка ваших данных (`news.txt`, `human.txt`).

## Установка

```bash
pip install -r requirements.txt
```

Если `faiss-cpu` не ставится через pip (часто на macOS), установите через conda.

## Быстрый старт (демо-данные)

Индексация:

```bash
python scripts/index.py \
  --data data/documents.jsonl \
  --index data/index.faiss \
  --meta data/index_meta.json
```

Поиск:

```bash
python scripts/search.py \
  --index data/index.faiss \
  --meta data/index_meta.json \
  --query "семантический поиск документов" \
  --top-k 5 \
  --show-text
```

Полезные фильтры:
- `--filter-source wikipedia`
- `--filter-category sport`
- `--filter-term энергия`
- `--score-threshold 0.35`

## Новости (`news.txt`)

Формат входного файла: `category<TAB>title<TAB>text`.

Конвертация в JSONL:

```bash
python scripts/prepare_news.py \
  --input news.txt \
  --output data/news.jsonl \
  --max-docs 500
```

Индексация и поиск (для русского — мультиязычная модель):

```bash
python scripts/index.py \
  --data data/news.jsonl \
  --index data/news.faiss \
  --meta data/news_meta.json \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

python scripts/search.py \
  --index data/news.faiss \
  --meta data/news_meta.json \
  --query "запасы нефти в США" \
  --top-k 5 \
  --show-text \
  --filter-category economics \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Обычный текст (`human.txt`)

Файл автоматически разбивается на чанки и сохраняется в JSONL:

```bash
python scripts/prepare_text.py \
  --input human.txt \
  --output data/human.jsonl \
  --title human \
  --max-chars 800

python scripts/index.py \
  --data data/human.jsonl \
  --index data/human.faiss \
  --meta data/human_meta.json \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

python scripts/search.py \
  --index data/human.faiss \
  --meta data/human_meta.json \
  --query "социальное значение расы" \
  --top-k 5 \
  --show-text \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Пример вывода:

```
Wrote 41 chunks to data/human.jsonl
Indexed 41 documents
Index saved to data/human.faiss
Metadata saved to data/human_meta.json
Query: социальное значение расы
[1] score=0.7960 | human (chunk 13/41) | source=text | chunk=13/41
    Согласно современным научным представлениям, раса — социальный конструкт...
[2] score=0.7372 | human (chunk 7/41) | source=text | chunk=7/41
    Будучи частично основанными на физическом сходстве людей внутри групп...
```

## Сравнение двух текстов (без индекса)

```bash
python scripts/search.py \
  --compare "нейросети для языка" "трансформеры в NLP"
```

## Файлы

- `data/documents.jsonl` — демо-коллекция
- `scripts/index.py` — индексация
- `scripts/search.py` — поиск
- `scripts/prepare_news.py` — конвертер `news.txt`
- `scripts/prepare_text.py` — конвертер обычного текста
- `requirements.txt` — зависимости
