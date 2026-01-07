from typing import List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

class AutocompleteTokenDataset(Dataset):
    """
    Датасет для автодополнения со скользящим окном.

    Для каждой позиции формируются пары (X, Y) где X - входная последовательность,
    Y - целевая последовательность для предсказания.

    Args:
        tokenized_texts: Список токенизированных текстов
        seq_len: Длина последовательности
        stride: Шаг для скользящего окна
    """

    def __init__(
        self,
        tokenized_texts: Sequence[List[int]],
        seq_len: int = 5,
        stride: int = 1,
    ) -> None:
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.samples: List[Tuple[List[int], List[int]]] = []

        for ids in tqdm(tokenized_texts, desc="Формирование (X, Y) со сдвигом"):
            n = len(ids)
            # Нужно минимум 2*seq_len - 1 токенов для полных X и Y
            min_needed = 2 * self.seq_len - 1
            if n < min_needed:
                continue

            # Последний допустимый старт i, чтобы конец Y не выходил за n
            last_start_inclusive = n - (2 * self.seq_len - 1)

            for i in range(0, last_start_inclusive + 1, self.stride):
                x_start = i
                x_end = i + self.seq_len
                y_start = x_start + 1
                y_end = y_start + self.seq_len

                # Здесь длины гарантированно ровные
                x_ids = ids[x_start:x_end]
                y_ids = ids[y_start:y_end]

                # Защитная проверка
                if len(x_ids) == self.seq_len and len(y_ids) == self.seq_len:
                    self.samples.append((x_ids, y_ids))

    def __len__(self) -> int:
        """Возвращает количество примеров в датасете."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает пару (X, Y) для обучения.

        Args:
            idx: Индекс примера

        Returns:
            Кортеж (input_tensor, target_tensor)
        """
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )