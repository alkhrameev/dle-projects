from utils import train
import torch


class Config:
    # для воспроизводимости
    SEED = 42

    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # Какие слои размораживаем - совпадают с нэймингом в моделях
    TEXT_MODEL_UNFREEZE = ""
    IMAGE_MODEL_UNFREEZE = ""

    # Гиперпараметры
    BATCH_SIZE = 256 
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    EPOCHS = 30
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    NUM_CLASSES = 4

    # Пути
    TRAIN_DF_PATH = "data/imdb_train.csv"
    VAL_DF_PATH = "data/imdb_val.csv"
    SAVE_PATH = "best_model.pth"


device = "cuda:1" if torch.cuda.is_available() else "cpu"
cfg = Config()

train(cfg, device)