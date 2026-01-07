from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class AutocompleteLSTM(nn.Module):
    """
    LSTM-модель для автодополнения текста.

    Предсказывает следующий токен на основе предыдущих. Входы имеют фиксированную длину.

    Args:
        vocab_size: Размер словаря
        embed_dim: Размерность эмбеддингов
        hidden_dim: Размерность скрытого состояния LSTM
        num_layers: Количество слоев LSTM
        dropout: Вероятность dropout
        pad_token_id: ID токена паддинга
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.lstm_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(hidden_dim, vocab_size)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Прямой проход модели.

        Args:
            input_ids: Тензор входных токенов [B, T]
            hidden: Скрытое состояние LSTM (опционально)

        Returns:
            logits: Логиты для предсказания следующего токена [B, T, vocab_size]
            hidden_out: Кортеж (h_n, c_n) - скрытое состояние LSTM
        """
        emb = self.embedding(input_ids)
        out, hidden = self.lstm(emb, hidden)
        out = self.lstm_dropout(out)
        logits = self.proj(out)
        return logits, hidden

    
    @torch.inference_mode()
    def generate(
        self,
        input_tokens: Union[torch.Tensor, list[int]],
        eos_token_id: Optional[int] = None,
        max_new_tokens: int = 20,
    ) -> list[int]:
        """
        Генерирует последовательность токенов на основе входных.

        Args:
            input_tokens: Входные токены [B, T] или список [T]
            eos_token_id: ID токена конца последовательности
            max_new_tokens: Максимальное количество новых токенов

        Returns:
            Список сгенерированных токенов
        """
        device = next(self.parameters()).device
        self.eval()

        # Преобразуем вход в тензор
        if isinstance(input_tokens, list):
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
        else:
            input_ids = input_tokens.to(device).long()
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # [T] -> [1, T]

        generated = []

        # Пропускаем входные токены целиком
        logits, hidden = self.forward(input_ids)
        next_logits = logits[:, -1, :]

        # Генерируем новые токены по одному, используя скрытое состояние
        for _ in range(max_new_tokens):
            next_token = torch.argmax(next_logits, dim=-1)
            tok_id = int(next_token[0].item())
            generated.append(tok_id)

            if eos_token_id is not None and tok_id == eos_token_id:
                break

            # Подаем следующий токен с сохранением hidden состояния
            next_token_tensor = next_token.unsqueeze(1)
            logits, hidden = self.forward(next_token_tensor, hidden)
            next_logits = logits[:, -1, :]

        return generated

    @torch.inference_mode()
    def generate_text(self, text, tokenizer, eos_token_id=None, max_new_tokens=20):
        """
        Генерирует текст на основе входной строки.

        Args:
            text: Входной текст
            tokenizer: Токенизатор для обработки
            eos_token_id: ID токена конца последовательности
            max_new_tokens: Максимальное количество новых токенов

        Returns:
            Сгенерированный текст
        """
        # Токенизируем входной текст
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        input_tokens = enc["input_ids"][0].tolist()
        
        # Генерируем токены
        generated_tokens = self.generate(
            input_tokens=input_tokens,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens
        )
        
        # Декодируем токены в текст
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()