import os
import argparse
import ast
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizerFast

from .next_token_dataset import AutocompleteTokenDataset
from .lstm_model import AutocompleteLSTM
from .data_utils import clean_csv_dataset, tokenize_and_split_dataset

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float = 3e-3,
    save_dir: str = "models",
    save_name: str = "best_autocomplete.pt",
    device=None,
    early_stopping_patience: int = 3,
    progress: bool = True,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
):
    """
    Обучает модель автодополнения с валидацией и early stopping.

    Args:
        model: Модель для обучения
        train_loader: DataLoader для обучающих данных
        val_loader: DataLoader для валидации
        epochs: Количество эпох
        lr: Learning rate
        save_dir: Директория для сохранения модели
        save_name: Имя файла модели
        device: Устройство (cuda/cpu)
        early_stopping_patience: Терпение для early stopping
        progress: Показывать прогресс-бар
        weight_decay: Коэффициент регуляризации
        grad_clip: Значение для gradient clipping

    Returns:
        Кортеж (модель, история обучения)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    model.to(device)

    # Оптимизатор и функция потерь
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Шедулер learning rate (уменьшает LR при плато)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=True,
    )

    # Подготовка сохранения
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        iterator = tqdm(loader, desc="Validating", leave=False) if progress else loader
        for x_batch, y_batch in iterator:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits, _ = model(x_batch, hidden=None)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            total_loss += loss.item()
            total_batches += 1
            if progress:
                iterator.set_postfix(loss=f"{total_loss / total_batches:.6f}")
        return total_loss / max(1, total_batches)

    # Основной цикл обучения
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True) if progress else train_loader

        for x_batch, y_batch in train_iter:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x_batch, hidden=None)

            loss = criterion(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if progress:
                current_lr = optimizer.param_groups[0]["lr"]
                train_iter.set_postfix(loss=f"{total_loss / total_batches:.6f}", lr=f"{current_lr:.2e}")

        train_loss = total_loss / max(1, total_batches)
        val_loss = evaluate(val_loader)

        # Шаг шедулера по валидации
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                save_path,
            )
            print(f"[train] Epoch {epoch:02d} | val_loss={val_loss:.6f} ✓ saved")
        else:
            epochs_no_improve += 1
            print(f"[train] Epoch {epoch:02d} | val_loss={val_loss:.6f}")

        print(f"[train] Epoch {epoch:02d} | train_loss={train_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if epochs_no_improve >= early_stopping_patience:
            print(f"[train] Early stopping after {early_stopping_patience} epochs without improvement.")
            break

    print(f"[train] Best val loss: {best_val_loss:.6f}. Saved to: {save_path}")
    return model, history


def prepare_dataloaders(
    train_data: List[List[int]],
    val_data: List[List[int]],
    test_data: List[List[int]],
    batch_size: int = 256,
):
    """
    Создает DataLoader'ы для обучения, валидации и теста.

    Args:
        train_data: Список токенизированных последовательностей для обучения
        val_data: Список токенизированных последовательностей для валидации
        test_data: Список токенизированных последовательностей для теста
        batch_size: Размер батча

    Returns:
        Кортеж (train_loader, val_loader, test_loader)
    """

    print(f"[prepare_dataloaders] train: {len(train_data)} | val: {len(val_data)} | test: {len(test_data)}")

    # Создаем датасеты для автодополнения
    print("[prepare_dataloaders] Создаем AutocompleteTokenDataset для train/val/test")
    train_ds = AutocompleteTokenDataset(train_data)
    val_ds = AutocompleteTokenDataset(val_data)
    test_ds = AutocompleteTokenDataset(test_data)
    print(f"[prepare_dataloaders] Длины AutocompleteTokenDataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Собираем DataLoader'ы с нужной функцией collation
    print(f"[prepare_dataloaders] Создаем DataLoader'ы с batch_size={batch_size}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print("[prepare_dataloaders] DataLoader'ы готовы к использованию")

    return train_loader, val_loader, test_loader


def load_tokenized_from_csv(csv_path: str) -> List[List[int]]:
    """
    Загружает токенизированные данные из CSV файла.

    Args:
        csv_path: Путь к CSV файлу с колонкой 'token_ids'

    Returns:
        Список токенизированных последовательностей
    """
    df = pd.read_csv(csv_path)
    if "token_ids" not in df.columns:
        raise ValueError(f"Столбец 'token_ids' не найден в {csv_path}")
    
    tokenized_texts = []
    for ids_str in df["token_ids"]:
        try:
            ids = ast.literal_eval(ids_str)
            if isinstance(ids, list) and all(isinstance(x, int) for x in ids):
                tokenized_texts.append(ids)
        except (ValueError, SyntaxError):
            continue
    
    return tokenized_texts


def main():
    parser = argparse.ArgumentParser(description="Обучение LSTM модели для автодополнения текста")
    
    parser.add_argument("--dataset-path", type=str, default=None, help="Путь к исходному CSV датасету")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    parser.add_argument("--device", type=str, default=None, help="Устройство (cuda/cpu)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Размер батча")
    parser.add_argument("--splits-dir", type=str, default="data/custom", help="Директория для сплитов")
    parser.add_argument("--save-dir", type=str, default="models", help="Директория для сохранения модели")
    
    args = parser.parse_args()
    
    # Загружаем токенизатор для получения vocab_size
    tokenizer_name = "bert-base-uncased"
    print(f"[main] Загружаем токенизатор: {tokenizer_name}")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    print(f"[main] Vocab size: {vocab_size}, Pad token ID: {pad_token_id}")
    
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"
    
    # Подготовка данных: очистка и сплит, если указан исходный датасет
    if args.dataset_path:
        print(f"[main] Подготовка данных")
        print(f"[main] Исходный датасет: {args.dataset_path}")
        
        # Очистка данных
        processed_path = splits_dir / "dataset_processed.csv"
        print(f"[main] Шаг 1: Очистка данных...")
        clean_csv_dataset(
            input_csv_path=args.dataset_path,
            output_csv_path=str(processed_path),
            text_column="text",
            min_words=3,
            max_words=50,
        )
        
        # Токенизация и сплит
        print(f"[main] Шаг 2: Токенизация и разбиение на сплиты...")
        train_data, val_data, test_data = tokenize_and_split_dataset(
            input_csv_path=str(processed_path),
            tokenizer=tokenizer,
            text_column="text",
            output_dir=str(splits_dir),
            random_state=42,
        )
        print(f"[main] Сплиты сохранены в {args.splits_dir}")
    else:
        # Загружаем готовые сплиты из CSV файлов
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Не найдены train.csv и/или val.csv в {args.splits_dir}. "
                f"Укажите --dataset-path для создания сплитов из исходного датасета."
            )
        
        print(f"[main] Загружаем готовые сплиты из {args.splits_dir}")
        train_data = load_tokenized_from_csv(str(train_path))
        val_data = load_tokenized_from_csv(str(val_path))
        test_data = load_tokenized_from_csv(str(test_path)) if test_path.exists() else []
    
    print(f"[main] Загружено: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    # Создаем даталоадеры
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=args.batch_size,
    )
    
    # Создаем модель
    print("[main] Создаем модель LSTM...")
    model = AutocompleteLSTM(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        pad_token_id=pad_token_id,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Модель создана. Параметров: {total_params:,} (обучаемых: {trainable_params:,})")
    
    # Запускаем обучение
    print("[main] Начинаем обучение...")
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        save_name="best_autocomplete.pt",
        device=args.device,
        early_stopping_patience=3,
        weight_decay=1e-2,
        grad_clip=args.grad_clip,
    )
    
    print("[main] Обучение завершено!")
    print(f"[main] Финальная история: train_loss={history['train_loss'][-1]:.6f}, val_loss={history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()