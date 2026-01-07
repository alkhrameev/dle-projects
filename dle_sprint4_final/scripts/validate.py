import torch
import numpy as np
from tqdm import tqdm


def validate(model, val_loader, device, mae_metric, pbar=None, target_mean=None, target_std=None):
    model.eval()
    
    iterator = pbar if pbar is not None else val_loader

    with torch.no_grad():
        for batch in iterator:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device) if batch['image'] is not None else None,
                'mass': batch['mass'].to(device)
            }
            calories_raw = batch['calories_raw'].to(device)

            logits = model(**inputs)
            # Денормализация предсказаний
            predicted_normalized = logits.squeeze(-1)
            predicted_denorm = predicted_normalized * target_std + target_mean
            _ = mae_metric(preds=predicted_denorm, target=calories_raw.float())
            
            if pbar is not None:
                current_mae = mae_metric.compute().cpu().numpy()
                pbar.set_postfix({'MAE': f'{current_mae:.4f}'})

    return mae_metric.compute().cpu().numpy()


def get_predictions_with_errors(model, val_loader, device, target_mean=None, target_std=None, show_progress=True):
    """
    Получает предсказания модели для всех блюд в валидационном датасете
    и вычисляет ошибки для каждого блюда.
    
    Args:
        model: Обученная модель
        val_loader: DataLoader для валидационного датасета
        device: Устройство (cuda/cpu)
        target_mean: Среднее значение для денормализации
        target_std: Стандартное отклонение для денормализации
        show_progress: Показывать ли прогресс-бар
    
    Returns:
        dict с ключами:
            - predictions: numpy array с предсказаниями
            - targets: numpy array с фактическими значениями
            - absolute_errors: numpy array с абсолютными ошибками
    """
    model.eval()
    all_predictions = []
    all_targets = []

    iterator = tqdm(val_loader, desc="Вычисление предсказаний") if show_progress else val_loader

    with torch.no_grad():
        for batch in iterator:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device) if batch['image'] is not None else None,
                'mass': batch['mass'].to(device)
            }
            calories_raw = batch['calories_raw'].to(device)

            logits = model(**inputs)
            # Денормализация предсказаний
            predicted_normalized = logits.squeeze(-1)
            predicted_denorm = predicted_normalized * target_std + target_mean

            all_predictions.append(predicted_denorm.cpu().numpy())
            all_targets.append(calories_raw.cpu().numpy())

    # Объединяем все предсказания и цели
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Вычисляем абсолютные ошибки
    absolute_errors = np.abs(all_predictions - all_targets)

    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'absolute_errors': absolute_errors
    }
