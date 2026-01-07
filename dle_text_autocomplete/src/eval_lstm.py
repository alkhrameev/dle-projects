import ast
import argparse
import time
import random
import pandas as pd
import evaluate
from tqdm import tqdm
import torch
from src.lstm_model import AutocompleteLSTM
from transformers import BertTokenizerFast


def eval_lstm(model, tokenizer, test_csv_path="data/custom/test.csv", device=None, eos_token_id=None):
    """
    Оценивает LSTM-модель по метрикам ROUGE-1 и ROUGE-2.

    Args:
        model: Модель с методом generate_text
        tokenizer: Токенизатор для декодирования
        test_csv_path: Путь к CSV с колонкой 'token_ids'
        device: Устройство для вычислений (cuda/cpu)
        eos_token_id: ID токена конца последовательности

    Returns:
        Словарь с метриками: rouge1, rouge2, count, time_sec
    """
    if eos_token_id is None:
        raise ValueError("[eval_lstm] Требуется указать параметр eos_token_id")

    start_time = time.time()
    print(f"[eval_lstm] Загрузка тестового датасета: {test_csv_path}")

    # Загружаем и парсим CSV
    df = pd.read_csv(test_csv_path)
    tokenized_sequences = []
    for s in df["token_ids"]:
        try:
            ids = ast.literal_eval(s)
            if isinstance(ids, list) and all(isinstance(x, int) for x in ids):
                tokenized_sequences.append(ids)
        except (ValueError, SyntaxError):
            continue

    print(f"[eval_lstm] Загружено {len(tokenized_sequences)} последовательностей")

    # Настраиваем модель
    model.eval()
    if device:
        model.to(device)
        print(f"[eval_lstm] Модель перенесена на устройство: {device}")

    rouge = evaluate.load("rouge")
    predictions, references = [], []

    with torch.no_grad():
        for tokens in tqdm(tokenized_sequences, desc="[eval_lstm] Оценка"):
            if len(tokens) < 4:
                continue

            # 3/4 - prompt, 1/4 - target
            split_idx = int(len(tokens) * 0.75)
            prompt_tokens = tokens[:split_idx]
            target_tokens = tokens[split_idx:]
            if not target_tokens:
                continue

            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True).strip()
            if not prompt_text:
                continue

            max_new_tokens = max(len(target_tokens) + 5, 5)

            gen_text = model.generate_text(
                text=prompt_text,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
            )

            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
            if not gen_text or not target_text:
                continue

            predictions.append(gen_text)
            references.append(target_text)

    # Подсчет ROUGE
    if predictions:
        scores = rouge.compute(predictions=predictions, references=references)
        rouge1 = float(scores.get("rouge1", 0.0))
        rouge2 = float(scores.get("rouge2", 0.0))
    else:
        rouge1 = rouge2 = 0.0

    elapsed = time.time() - start_time
    results = {"rouge1": rouge1, "rouge2": rouge2, "count": len(predictions), "time_sec": elapsed}

    # Итоговый вывод
    print("\n[eval_lstm] РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("[eval_lstm] ================================")
    print(f"[eval_lstm] ROUGE-1: {rouge1:.4f}")
    print(f"[eval_lstm] ROUGE-2: {rouge2:.4f}")
    print(f"[eval_lstm] Количество примеров: {len(predictions)}")
    print(f"[eval_lstm] Время выполнения: {elapsed:.1f} сек.")
    print("[eval_lstm] ================================\n")

    # Показываем 5 случайных примеров
    if predictions:
        print("[eval_lstm] Примеры генерации (5 случайных):")
        print("[eval_lstm] --------------------------------")
        sample_indices = random.sample(range(len(predictions)), min(5, len(predictions)))
        for i, idx in enumerate(sample_indices, 1):
            print(f"[{i}]")
            print(f"[eval_lstm]  ➤ Сгенерировано: {predictions[idx]}")
            print(f"[eval_lstm]  ✔ Оригинал:     {references[idx]}")
            print("[eval_lstm] --------------------------------")

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка генерации HuggingFace transformer pipeline по ROUGE-1/2.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    vocab_size = getattr(tokenizer, "vocab_size", None)
    eos_token_id = tokenizer.sep_token_id 

    lstm = AutocompleteLSTM(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=128,
            num_layers=1,
            dropout=0.3,
            pad_token_id=pad_token_id,
    )

    checkpoint = torch.load("models/best_autocomplete.pt", map_location="cuda:1")
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm.eval()

    results = eval_lstm(
        model=lstm,
        tokenizer=tokenizer,
        test_csv_path="data/custom/test.csv",
        eos_token_id = eos_token_id,
        device=args.device
    )

    print(f"[main] Результаты: {results}")