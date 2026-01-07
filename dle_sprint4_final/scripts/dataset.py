import albumentations as A
import numpy as np
import pandas as pd
import torch
import timm
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def compute_target_stats(train_df_path):
    """Вычисляет mean и std целевой переменной на train данных"""
    train_df = pd.read_csv(train_df_path)
    mean = train_df['total_calories'].mean()
    std = train_df['total_calories'].std()
    return mean, std


def compute_mass_stats(train_df_path):
    """Вычисляет mean и std total_mass на train данных"""
    train_df = pd.read_csv(train_df_path)
    mean = train_df['total_mass'].mean()
    std = train_df['total_mass'].std()
    return mean, std


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train", target_mean=None, target_std=None, mass_mean=None, mass_std=None):
        if ds_type == "train":
            self.df = pd.read_csv(config.TRAIN_DF_PATH)
        else:
            self.df = pd.read_csv(config.VAL_DF_PATH)
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms
        
        # Нормализация целевой переменной
        if target_mean is None or target_std is None:
            # Вычисляем статистику на train данных
            self.target_mean, self.target_std = compute_target_stats(config.TRAIN_DF_PATH)
        else:
            self.target_mean = target_mean
            self.target_std = target_std
        
        # Нормализация total_mass
        if mass_mean is None or mass_std is None:
            # Вычисляем статистику на train данных
            self.mass_mean, self.mass_std = compute_mass_stats(config.TRAIN_DF_PATH)
        else:
            self.mass_mean = mass_mean
            self.mass_std = mass_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "ingredients_names"]
        calories = self.df.loc[idx, "total_calories"]
        mass = self.df.loc[idx, "total_mass"]
        
        # Нормализация целевой переменной
        calories_normalized = (calories - self.target_mean) / self.target_std
        
        # Нормализация total_mass
        mass_normalized = (mass - self.mass_mean) / self.mass_std

        dish_id = self.df.loc[idx, "dish_id"]
        img_path = f"data/images/{dish_id}/rgb.png"
        
        image = None
        try:
            img = Image.open(img_path).convert('RGB')
            image = self.transforms(image=np.array(img))["image"]
        except:
            # Если изображение отсутствует, image останется None
            pass

        return {
            "calories": calories_normalized, 
            "calories_raw": calories,  # Сохраняем оригинальное значение для денормализации
            "mass": mass_normalized,
            "image": image, 
            "text": text
        }


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    calories = torch.FloatTensor([item["calories"] for item in batch])  # Нормализованные значения
    calories_raw = torch.FloatTensor([item["calories_raw"] for item in batch])  # Оригинальные значения
    mass = torch.FloatTensor([item["mass"] for item in batch])  # Нормализованные значения массы

    # Собираем изображения
    images_list = []
    first_real_img = None
    
    # Находим первое реальное изображение для определения размера
    for item in batch:
        if item["image"] is not None:
            if first_real_img is None:
                first_real_img = item["image"]
            images_list.append(item["image"])
        else:
            images_list.append(None)
    
    # Если есть хотя бы одно реальное изображение, создаем тензор
    if first_real_img is not None:
        # Заменяем None на нулевые тензоры того же размера
        images = []
        for img in images_list:
            if img is not None:
                images.append(img)
            else:
                images.append(torch.zeros_like(first_real_img))
        images = torch.stack(images)
    else:
        # Если нет ни одного изображения
        images = None

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "calories": calories,  # Нормализованные значения для обучения
        "calories_raw": calories_raw,  # Оригинальные значения для метрик
        "mass": mass,  # Нормализованные значения массы
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        # Используем среднее значение для fill (обычно 128 для RGB изображений)
        fill_value = 128
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]) + 32, p=1.0),
                A.RandomCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=fill_value,
                         p=0.8),
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(int(0.07 * cfg.input_size[1]),
                                       int(0.15 * cfg.input_size[1])),
                    hole_width_range=(int(0.1 * cfg.input_size[2]),
                                      int(0.15 * cfg.input_size[2])),
                    fill=fill_value,
                    p=0.5),
                A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.1,
                              p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )

    return transforms


def get_train_loader(config, tokenizer=None):
    """
    Создает train DataLoader с необходимыми трансформациями и нормализацией.
    
    Args:
        config: Конфигурация с параметрами модели и путей к данным
        tokenizer: Токенизатор для текста. Если None, создается автоматически.
    
    Returns:
        tuple: (train_loader, target_mean, target_std, mass_mean, mass_std)
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Вычисляем статистику для нормализации на train данных
    target_mean, target_std = compute_target_stats(config.TRAIN_DF_PATH)
    mass_mean, mass_std = compute_mass_stats(config.TRAIN_DF_PATH)
    
    # Создаем трансформации для train
    transforms = get_transforms(config, ds_type="train")
    
    # Создаем train датасет
    train_dataset = MultimodalDataset(
        config, transforms, ds_type="train",
        target_mean=target_mean, target_std=target_std,
        mass_mean=mass_mean, mass_std=mass_std
    )
    
    # Создаем train DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    
    return train_loader, target_mean, target_std, mass_mean, mass_std


def get_val_loader(config, tokenizer=None, target_mean=None, target_std=None, 
                   mass_mean=None, mass_std=None):
    """
    Создает validation DataLoader с необходимыми трансформациями и нормализацией.
    
    Args:
        config: Конфигурация с параметрами модели и путей к данным
        tokenizer: Токенизатор для текста. Если None, создается автоматически.
        target_mean: Среднее значение для нормализации калорий. Если None, вычисляется.
        target_std: Стандартное отклонение для нормализации калорий. Если None, вычисляется.
        mass_mean: Среднее значение для нормализации массы. Если None, вычисляется.
        mass_std: Стандартное отклонение для нормализации массы. Если None, вычисляется.
    
    Returns:
        tuple: (val_loader, target_mean, target_std, mass_mean, mass_std)
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    
    # Вычисляем статистику для нормализации, если не передана
    if target_mean is None or target_std is None:
        target_mean, target_std = compute_target_stats(config.TRAIN_DF_PATH)
    if mass_mean is None or mass_std is None:
        mass_mean, mass_std = compute_mass_stats(config.TRAIN_DF_PATH)
    
    # Создаем трансформации для validation
    val_transforms = get_transforms(config, ds_type="val")
    
    # Создаем validation датасет
    val_dataset = MultimodalDataset(
        config, val_transforms, ds_type="val",
        target_mean=target_mean, target_std=target_std,
        mass_mean=mass_mean, mass_std=mass_std
    )
    
    # Создаем validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    
    return val_loader, target_mean, target_std, mass_mean, mass_std
