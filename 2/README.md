# Семантический grep (вариант 2)

Утилита ищет строки в текстовом файле не только по точному слову, но и по словам, близким по смыслу, используя Word2Vec.

## Файлы

- `train_word2vec.py` — обучение модели Word2Vec на корпусе новостей.
- `mygrep.py` — поиск строк с учетом синонимов.

## Зависимости

- Python 3.10+
- `gensim` (и совместимый `numpy`)
- `pymorphy2` или `pymorphy3` (опционально, для лемматизации)

Минимальная установка:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install gensim numpy
```

## Обучение модели

По умолчанию скрипт ищет корпус в `nlp-2025/data/news.txt.gz` и сохраняет модель в `models/word2vec.model`.
Если установлен `pymorphy2` или `pymorphy3`, токены лемматизируются перед обучением.

```bash
python3 train_word2vec.py
```

## Поиск

```bash
python3 mygrep.py data.txt "привет" --model models/news.model --topn 10 --threshold 0.55 --show-terms
```

Вывод всегда идет нумерованным списком.

Если файл в формате `категория<TAB>заголовок<TAB>текст`, можно выводить только нужные части:

```bash
python3 mygrep.py data.txt "привет" --model models/news.model --fields category_headline
python3 mygrep.py data.txt "привет" --model models/news.model --fields text --max-len 160
```

Если нужно выводить только предложение, где встречается слово:

```bash
python3 mygrep.py data.txt "привет" --model models/news.model --sentence
python3 mygrep.py data.txt "привет" --model models/news.model --sentence --max-len 160
```

В режиме `--sentence` в конце каждого предложения показывается слово,
которое сработало как синоним (или точное совпадение).

Вывести только список синонимов:

```bash
python3 mygrep.py data.txt "движение" --model models/news.model --only-synonyms
```

Для проверки «подстановочной адекватности» используется контекстное
согласование на Word2Vec (без контекстных моделей). Фильтр можно настроить:

```bash
python3 mygrep.py data.txt "движение" --model models/news.model --sentence \
  --context-window 5 --context-threshold 0.3 --context-source-threshold 0.2 \
  --context-delta 0.3 --context-min-tokens 2
```

По умолчанию синонимы берутся только из той же письменности, что и запрос
(например, для «спорт» будут отфильтрованы латиница и доменные части вроде `ua`).
Если это ограничение мешает, добавьте `--cross-script`.

По умолчанию используется лемматизация (требуется `pymorphy2` или `pymorphy3`).
Если библиотека не установлена, лемматизация автоматически выключится с предупреждением.
Отключить вручную можно так:

```bash
python3 mygrep.py data.txt "привет" --model models/news.model --no-lemma
```

Для редких слов эмбеддинг нестабилен, поэтому по умолчанию синонимы
не расширяются, если слово встречалось в корпусе меньше 10 раз. Это можно изменить:

```bash
python3 mygrep.py data.txt "привет" --model models/news.model --min-freq 1
```

По умолчанию используется «взаимная близость»: слово считается синонимом,
только если запрос входит в топ похожих слов для кандидата. Отключить можно так:

```bash
python3 mygrep.py data.txt "привет" --model models/news.model --mutual-topn 0
```

Если нужно искать только точные совпадения (без модели):

```bash
python3 mygrep.py data.txt "привет" --exact
```

## Поведение

- Поиск идет по токенам (регистр не учитывается).
- При наличии `pymorphy2`/`pymorphy3` сравнение идет по леммам.
- Если слово отсутствует в словаре (OOV) или синонимы не найдены, остается только точное совпадение.
- Поддерживаются файлы `.gz`.
- Ошибки чтения файла и загрузки модели выводятся в `stderr`.
