import os
import re
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
import emoji
from cleantext import clean

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split


def download_tweets_dataset(output_csv_path: str,
                           source_url: str = "https://code.s3.yandex.net/deep-learning/tweets.txt") -> str:
    """
    Скачивает твиты по URL и сохраняет в CSV.

    Args:
        output_csv_path: Путь для сохранения CSV файла
        source_url: URL источника данных

    Returns:
        Путь к сохраненному файлу
    """
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[download_tweets_dataset] Скачивание датасета твитов с удаленного источника ...")
    resp = requests.get(source_url, timeout=60)
    resp.raise_for_status()
    # Гарантируем Unix-переводы строк и очистку пустых
    lines = [line.strip() for line in resp.text.splitlines() if line.strip()]

    df = pd.DataFrame({"text": lines})
    print(f"[download_tweets_dataset] Загружено {len(df)} строк.")
    df.to_csv(output_path, index=False)
    print(f"[download_tweets_dataset] Сохранено в {output_path}")
    return str(output_path)


def clean_text(text: str, lower: bool = True) -> str:
    """
    Очищает текст от мусора и нормализует его.

    Args:
        text: Исходный текст
        lower: Привести к нижнему регистру

    Returns:
        Очищенный текст
    """
    # Удаляем упоминания пользователей с символом @
    text_no_mentions = re.sub(r'@\w+', '', text)
    cleaned = clean(
        text_no_mentions,
        to_ascii=True,          
        lower=lower,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_line_breaks=True,
        no_numbers=True,
        no_digits=True,
        no_currency_symbols=True,
        no_punct=True,
        replace_with_url="",    
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="",
        replace_with_currency_symbol="",
    ).strip()
    return cleaned


def clean_csv_dataset(input_csv_path: str,
                      output_csv_path: str,
                      text_column: str = "text",
                      drop_empty: bool = True,
                      min_words: int = 3,
                      max_words: int = 50) -> str:
    """
    Очищает CSV датасет: нормализует текст, разбивает на предложения, фильтрует по длине.

    Args:
        input_csv_path: Путь к входному CSV
        output_csv_path: Путь для сохранения результата
        text_column: Название столбца с текстом
        drop_empty: Удалять пустые строки и дубликаты
        min_words: Минимальное количество слов в предложении
        max_words: Максимальное количество слов в предложении

    Returns:
        Путь к сохраненному файлу
    """

    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[clean_csv_dataset] Чистка столбца '{text_column}' в {input_path} ...")
    df = pd.read_csv(input_path)

    if text_column not in df.columns:
        raise ValueError(f"Столбец '{text_column}' не найден в {input_csv_path}")

    # Один эффективный фильтрующий проход, предложение из строки
    cleaned_texts = []
    orig_total = len(df)
    removed_too_short = 0
    removed_too_long = 0
    removed_empty = 0

    def split_into_sentences(text):
        # Разбиваем на предложения по [.?!], убираем короткие (<2 символа)
        sentence_end_re = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_end_re.split(text)
        return [s.strip() for s in sentences if len(s.strip()) > 1]

    for text in tqdm(df[text_column], desc="Чистка, сплит на предложения, фильтрация"):
        txt_clean = clean_text(text)
        # Разбиваем на предложения
        for sent in split_into_sentences(txt_clean):
            word_count = len(sent.split())
            if not sent.strip():
                removed_empty += 1
                continue
            if word_count < min_words:
                removed_too_short += 1
                continue
            if word_count > max_words:
                removed_too_long += 1
                continue
            cleaned_texts.append(sent)

    print(f"[clean_csv_dataset] Удалено {removed_too_short} предложений (<{min_words} слов), {removed_too_long} предложений (>{max_words} слов), {removed_empty} пустых.")
    result_df = pd.DataFrame({text_column: cleaned_texts})

    if drop_empty:
        before = len(result_df)
        result_df[text_column] = result_df[text_column].str.strip()
        result_df = result_df[result_df[text_column].astype(bool)]
        result_df = result_df.drop_duplicates(subset=[text_column])
        dropped = before - len(result_df)
        if dropped > 0:
            print(f"[clean_csv_dataset] Удалено {dropped} дублей/пустых строк.")

    result_df.to_csv(output_path, index=False)
    print(f"[clean_csv_dataset] Сохранено в {output_path}")
    return str(output_path)


def tokenize_and_split_dataset(
    input_csv_path: str,
    tokenizer,
    text_column: str = "text",
                      add_special_tokens: bool = True,
                      output_dir: Optional[str] = None,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Токенизирует датасет и делит на train/val/test.

    Args:
        input_csv_path: Путь к входному CSV
        tokenizer: Токенизатор для обработки текста
        text_column: Название столбца с текстом
        add_special_tokens: Добавлять специальные токены
        output_dir: Директория для сохранения сплитов (опционально)
        test_size: Доля данных для val+test (по умолчанию 0.2)
        random_state: Seed для случайного разбиения

    Returns:
        Кортеж (train_data, val_data, test_data) - списки токенизированных последовательностей
    """

    p = Path(input_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Файл {input_csv_path} не найден")

    print(f"[tokenize_and_split_dataset] Загружаем датасет из {p} ...")
    df = pd.read_csv(p)
    if text_column not in df.columns:
        raise ValueError(f"Столбец '{text_column}' не найден в {input_csv_path}")

    # Фильтруем пустые строки
    texts = (
        df[text_column]
        .astype(str)
        .map(lambda s: s.strip())
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )

    print(f"[tokenize_and_split_dataset] Всего текстов: {len(texts)}. Начинаем токенизацию...")

    tokenized = []
    filtered_texts = []

    for t in tqdm(texts, desc="[tokenize_and_split_dataset] Токенизация"):
        token_ids = tokenizer.encode(t, add_special_tokens=add_special_tokens)
        # Пропускаем пустые последовательности
        if token_ids:
            tokenized.append(token_ids)
            filtered_texts.append(t)

    if not tokenized:
        print("[tokenize_and_split_dataset] Нет токенизированных данных (все тексты пустые).")
        return [], [], []

    print(f"[tokenize_and_split_dataset] Токенизировано {len(tokenized)} текстов.")

    # Делим: train vs (val+test)
    train_pairs, rest_pairs = train_test_split(
        list(zip(filtered_texts, tokenized)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    # Val и test поровну
    val_pairs, test_pairs = train_test_split(
        rest_pairs,
        test_size=0.5,
        random_state=random_state,
        shuffle=True
    )

    train_texts, train_data = zip(*train_pairs) if train_pairs else ([], [])
    val_texts, val_data = zip(*val_pairs) if val_pairs else ([], [])
    test_texts, test_data = zip(*test_pairs) if test_pairs else ([], [])

    print(f"[tokenize_and_split_dataset] Разбиение: train: {len(train_data)} текстов, val: {len(val_data)} текстов, test: {len(test_data)} текстов")

    # При необходимости сохраняем
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def save_split(name: str, texts: List[str], tokens: List[List[int]]):
            pd.DataFrame({
                "text": texts,
                "token_ids": [str(x) for x in tokens]
            }).to_csv(out / f"{name}.csv", index=False)
            print(f"[tokenize_and_split_dataset] Сохранено {len(texts)} строк в {out / f'{name}.csv'}")

        save_split("train", train_texts, train_data)
        save_split("val", val_texts, val_data)
        save_split("test", test_texts, test_data)

    return list(train_data), list(val_data), list(test_data)