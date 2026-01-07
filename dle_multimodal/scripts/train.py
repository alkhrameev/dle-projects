import torch
import torch.nn as nn
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
from tqdm import tqdm

from .dataset import get_train_loader, get_val_loader
from .model import MultimodalModel
from .utils import seed_everything, set_requires_grad
from .validate import validate


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.classifier.parameters(),
        'lr': config.CLASSIFIER_LR
    }])

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=getattr(config, 'LR_FACTOR', 0.5),
        patience=getattr(config, 'LR_PATIENCE', 3),
        verbose=True,
        min_lr=getattr(config, 'MIN_LR', 1e-7)
    )

    criterion = nn.MSELoss()  # MSE loss для регрессии
    
    # Early stopping
    early_stopping_patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 5)
    early_stopping_counter = 0

    # Загрузка данных
    train_loader, target_mean, target_std, mass_mean, mass_std = get_train_loader(config, tokenizer)
    val_loader, _, _, _, _ = get_val_loader(
        config, tokenizer, 
        target_mean=target_mean, target_std=target_std,
        mass_mean=mass_mean, mass_std=mass_std
    )
    
    print(f"Target normalization: mean={target_mean:.2f}, std={target_std:.2f}")
    print(f"Mass normalization: mean={mass_mean:.2f}, std={mass_std:.2f}")

    # инициализируем метрику
    mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_metric_val = torchmetrics.MeanAbsoluteError().to(device)
    # best_mae_train = float('inf')
    best_mae_val = float('inf')

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for batch in train_pbar:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device) if batch['image'] is not None else None,
                'mass': batch['mass'].to(device)
            }
            calories = batch['calories'].to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits.squeeze(-1), calories.float())

            # Backward
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = getattr(config, 'MAX_GRAD_NORM', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            total_loss += loss.item()

            # Денормализация для вычисления метрик
            predicted_normalized = logits.squeeze(-1)
            predicted_denorm = predicted_normalized * target_std + target_mean
            calories_raw = batch['calories_raw'].to(device)
            _ = mae_metric_train(preds=predicted_denorm, target=calories_raw.float())
            
            # Обновляем прогресс-бар
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/len(train_pbar):.4f}'
            })

        # Валидация
        train_mae = mae_metric_train.compute().cpu().numpy()
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]")
        val_mae = validate(model, val_loader, device, mae_metric_val, val_pbar, target_mean, target_std)
        mae_metric_val.reset()
        mae_metric_train.reset()

        # Обновление learning rate scheduler
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | avg_Loss: {total_loss/len(train_loader):.4f} | Train MAE: {train_mae :.4f}| Val MAE: {val_mae :.4f} | LR: {current_lr:.2e}"
        )

        if val_mae < best_mae_val:
            print(f"New best model, epoch: {epoch}, Val MAE: {val_mae:.4f}")
            best_mae_val = val_mae
            early_stopping_counter = 0
            torch.save(model.state_dict(), config.SAVE_PATH)
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epochs. Best Val MAE: {best_mae_val:.4f}")
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best Val MAE: {best_mae_val:.4f}")
                break