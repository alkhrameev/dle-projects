import ast
import time
import random
import argparse
import pandas as pd
import evaluate
from tqdm import tqdm
from transformers import pipeline

def eval_transformer_pipeline(
    model_name="distilgpt2",
    test_csv_path="data/custom/test.csv",
    device=None,
    do_sample=True,
    top_k=50,
    temperature=1.0,
    val_fraction=1.0,
):
    """
    Оценивает HuggingFace transformer модель по метрикам ROUGE-1/2.

    Args:
        model_name: Название модели из HuggingFace
        test_csv_path: Путь к CSV с колонкой 'token_ids'
        device: Устройство для вычислений
        do_sample: Использовать сэмплирование при генерации
        top_k: Параметр top-k sampling
        temperature: Температура для сэмплирования
        val_fraction: Доля датасета для валидации (0 < val_fraction <= 1.0)

    Returns:
        Словарь с метриками: rouge1, rouge2, count, time_sec
    """
    start_time = time.time()
    print(f"[eval_transformer_pipeline] Загрузка тестового датасета: {test_csv_path}")

    # Загрузка и парсинг CSV
    df = pd.read_csv(test_csv_path)
    tokenized_sequences = []
    for s in df["token_ids"]:
        try:
            seq = ast.literal_eval(s)
            if isinstance(seq, list) and all(isinstance(x, int) for x in seq):
                tokenized_sequences.append(seq)
        except Exception:
            continue
    print(f"[eval_transformer_pipeline] Загружено {len(tokenized_sequences)} последовательностей")

    # Берем только часть датасета (перемешиваем и выбираем долю)
    random.shuffle(tokenized_sequences)
    n = max(1, int(len(tokenized_sequences) * val_fraction))
    tokenized_sequences = tokenized_sequences[:n]
    print(f"[eval_transformer_pipeline] Валидация будет по случайной подвыборке: {len(tokenized_sequences)} примеров ({val_fraction:.2%})")

    # Инициализация pipeline
    print(f"[eval_transformer_pipeline] Инициализация pipeline с моделью: {model_name}")
    generator = pipeline("text-generation", model=model_name, device=device)
    tokenizer = generator.tokenizer
    print(f"[eval_transformer_pipeline] Pipeline инициализирован (device: {device})")

    rouge = evaluate.load("rouge")
    predictions, references = [], []

    for tokens in tqdm(tokenized_sequences, desc="[eval_transformer_pipeline] Оценка"):
        if len(tokens) < 4:
            continue

        split_idx = int(len(tokens) * 0.75)
        prompt_tokens, target_tokens = tokens[:split_idx], tokens[split_idx:]
        if not target_tokens:
            continue

        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True).strip()
        if not prompt_text:
            continue

        # Динамический max_new_tokens
        dynamic_max_new_tokens = max(1, len(target_tokens))

        gen_kwargs = {
            "max_new_tokens": dynamic_max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": 1,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"top_k": top_k, "temperature": temperature})

        result = generator(prompt_text, **gen_kwargs)
        generated_text = result[0]["generated_text"]
        gen_text = (
            generated_text[len(prompt_text):].strip()
            if generated_text.startswith(prompt_text)
            else generated_text.strip()
        )

        target_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
        if gen_text and target_text:
            predictions.append(gen_text)
            references.append(target_text)

    # Подсчет ROUGE
    scores = rouge.compute(predictions=predictions, references=references) if predictions else {}
    rouge1 = float(scores.get("rouge1", 0.0))
    rouge2 = float(scores.get("rouge2", 0.0))
    elapsed = time.time() - start_time

    # Итоговый вывод
    print("\n[eval_transformer_pipeline] РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("[eval_transformer_pipeline] ================================")
    print(f"[eval_transformer_pipeline] ROUGE-1: {rouge1:.4f}")
    print(f"[eval_transformer_pipeline] ROUGE-2: {rouge2:.4f}")
    print(f"[eval_transformer_pipeline] Количество примеров: {len(predictions)}")
    print(f"[eval_transformer_pipeline] Время выполнения: {elapsed:.1f} сек.")
    print("[eval_transformer_pipeline] ================================\n")

    # Примеры
    if predictions:
        print("[eval_transformer_pipeline] Примеры генерации (5 случайных):")
        print("[eval_transformer_pipeline] --------------------------------")
        for i, idx in enumerate(random.sample(range(len(predictions)), min(5, len(predictions))), 1):
            print(f"[{i}]")
            print(f"[eval_transformer_pipeline]  ➤ Сгенерировано: {predictions[idx]}")
            print(f"[eval_transformer_pipeline]  ✔ Оригинал:     {references[idx]}")
            print("[eval_transformer_pipeline] --------------------------------")

    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "count": len(predictions),
        "time_sec": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка генерации HuggingFace transformer pipeline по ROUGE-1/2.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=1.0,
        help="Доля от датасета для валидации (от 0 до 1, например, 0.2 = 20%%)",
    )
    args = parser.parse_args()

    results = eval_transformer_pipeline(device=args.device, val_fraction=args.val_fraction)

    print(f"[main] Результаты: {results}")