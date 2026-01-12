# Семантический grep (вариант 2)

Утилита ищет строки или предложения по слову и его синонимам. Синонимы берутся из Word2Vec; опционально можно включить фильтр по RuWordNet.

## Файлы

- `train_word2vec.py` — обучение модели Word2Vec.
- `mygrep.py` — поиск по файлу с учетом синонимов.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install gensim numpy
```

Дополнительно для словаря:

```bash
pip install ruwordnet
ruwordnet download
```

Если нужен учет форм слова для словаря:

```bash
pip install pymorphy3
```

## Обучение модели

```bash
python3 train_word2vec.py --corpus news.txt --model models/news.model
```

По умолчанию: `nlp-2025/data/news.txt.gz` -> `models/word2vec.model`.

## Поиск

```bash
python3 mygrep.py news.txt "движение" --model models/news.model --sentence --max-len 160
```

Если файл в формате `категория<TAB>заголовок<TAB>текст`:

```bash
python3 mygrep.py news.txt "движение" --model models/news.model --fields text
```

По умолчанию выводится только предложение; полный текст строки — `--full-line`.
В конце предложения печатается `(совпадение: ...)` или `(синоним: ...)`.

## Пример

Команда:

```bash
python3 mygrep.py news.txt "движение" --model models/news.model --only-synonyms --lexicon
```

Пример вывода:

```
передвижение
перемещение
```

## Синонимы

- Word2Vec дает кандидатов (`--topn`, `--threshold`, `--mutual-topn`), по умолчанию `100 / 0.3 / 0`.
- `--lexicon` включает RuWordNet; синонимы = пересечение Word2Vec и RuWordNet.
- Если установлен `pymorphy3`, словарь работает по леммам: формы одной леммы считаются точным совпадением, а синонимами — только другие леммы.

Показать только синонимы:

```bash
python3 mygrep.py news.txt "движение" --model models/news.model --only-synonyms
python3 mygrep.py news.txt "движение" --model models/news.model --only-synonyms --lexicon
```

## Полезные параметры

- `--topn` — сколько кандидатов брать из Word2Vec (по умолчанию 100).
- `--threshold` — минимум похожести; ниже = больше слов (по умолчанию 0.3).
- `--mutual-topn` — взаимность; `0` отключает (по умолчанию 0).
- `--min-freq` — не расширять редкие слова (по умолчанию 1).
- `--context-*` — контекстная проверка «подстановочной адекватности».
- `--cross-script` — разрешить синонимы в другой письменности.
- `--exact` — искать только точные совпадения без модели.

## Примечания

- Если синонимы не найдены, поиск идет по точному слову.
- Поддерживаются файлы `.gz`.
