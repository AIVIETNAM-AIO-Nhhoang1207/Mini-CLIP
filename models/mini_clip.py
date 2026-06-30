import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import DistilBertModel


class ImageEncoder(nn.Module):
    """Image Encoder: ResNet-18 backbone + Linear projection."""

    def __init__(self, embed_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(512, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embed = self.projection(features)
        return embed


class TextEncoder(nn.Module):
    """Text Encoder: DistilBERT backbone + Linear projection."""

    def __init__(self, model_name="distilbert-base-uncased", embed_dim=512):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
        self.projection = nn.Linear(768, embed_dim)
        self.target_token_idx = 0  # [CLS] token

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        cls_output = last_hidden_state[:, self.target_token_idx, :]
        embed = self.projection(cls_output)
        return embed


class MiniCLIP(nn.Module):
    """
    Mini-CLIP: Contrastive Language-Image Pre-training.
    Kết hợp ImageEncoder (ResNet-18) + TextEncoder (DistilBERT).
    """

    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        return image_features, text_features, self.logit_scale.exp()


def contrastive_loss(image_features, text_features, logit_scale):
    """Symmetric contrastive loss (InfoNCE)."""
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    labels = torch.arange(image_features.shape[0], device=image_features.device)

    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)

    return (loss_img + loss_txt) / 2
