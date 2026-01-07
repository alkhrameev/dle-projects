import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM)

        # Классификатор принимает умноженные фичи: text * image * mass = HIDDEN_DIM
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2), #bottleneck - заставляет модель выделять наиболее значимые признаки, уменьшает вероятность переобучения
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        text_emb = self.text_proj(text_features)
        
        # Обработка изображения
        if image is not None:
            image_features = self.image_model(image)
            image_emb = self.image_proj(image_features)
        else:
            # Если нет изображений, используем нулевой вектор для image_emb
            batch_size = text_emb.size(0)
            image_emb = torch.zeros(batch_size, self.hidden_dim, device=text_emb.device)
        
        # Обработка total_mass
        mass_emb = self.mass_proj(mass.unsqueeze(-1))
        
        # Умножение фичей (text * image * mass)
        fused_emb = text_emb * image_emb * mass_emb

        logits = self.classifier(fused_emb)
        return logits
