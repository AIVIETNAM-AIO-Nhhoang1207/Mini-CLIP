import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, BertModel


# ============================================================
#  Image Encoders
# ============================================================

class ImageEncoder(nn.Module):
    """Image encoder dùng ResNet-18 (baseline)."""
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


class ImageEncoderViT(nn.Module):
    """Image encoder dùng ViT-B/16."""
    def __init__(self, embed_dim=512):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.backbone = vit
        # ViT-B/16 output 768-dim ở head
        # Thay head bằng Identity để lấy feature thô
        self.backbone.heads = nn.Identity()
        self.projection = nn.Linear(768, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        embed = self.projection(features)
        return embed


# ============================================================
#  Text Encoders
# ============================================================

class TextEncoder(nn.Module):
    """Text encoder dùng DistilBERT (baseline)."""
    def __init__(self, model_name="distilbert-base-uncased", embed_dim=512):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
            
        self.projection = nn.Linear(768, embed_dim)
        
        self.target_token_idx = 0 

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state 
        cls_output = last_hidden_state[:, self.target_token_idx, :]
        embed = self.projection(cls_output)
        
        return embed


class TextEncoderBERT(nn.Module):
    """Text encoder dùng BERT-base."""
    def __init__(self, model_name="bert-base-uncased", embed_dim=512):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        # BERT-base output 768-dim
        self.projection = nn.Linear(768, embed_dim)
        self.target_token_idx = 0  # CLS token

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        cls_output = last_hidden_state[:, self.target_token_idx, :]
        embed = self.projection(cls_output)
        return embed


# ============================================================
#  Factory Function
# ============================================================

# Danh sách tất cả configs
ALL_CONFIGS = [
    ("resnet18", "distilbert"),
    ("vit",      "distilbert"),
    ("resnet18", "bert"),
]


def get_encoders(image_type="resnet18", text_type="distilbert", embed_dim=512):
    """
    Factory function tạo cặp (image_encoder, text_encoder, tokenizer_name).
    
    Args:
        image_type: "resnet18" hoặc "vit"
        text_type:  "distilbert" hoặc "bert"
        embed_dim:  chiều embedding output
    
    Returns:
        (image_encoder, text_encoder, tokenizer_name)
    """
    # Image encoder
    if image_type == "resnet18":
        img_enc = ImageEncoder(embed_dim)
    elif image_type == "vit":
        img_enc = ImageEncoderViT(embed_dim)
    else:
        raise ValueError(f"Unknown image encoder: {image_type}. Chọn: resnet18, vit")

    # Text encoder
    if text_type == "distilbert":
        txt_enc = TextEncoder(embed_dim=embed_dim)
        tokenizer_name = "distilbert-base-uncased"
    elif text_type == "bert":
        txt_enc = TextEncoderBERT(embed_dim=embed_dim)
        tokenizer_name = "bert-base-uncased"
    else:
        raise ValueError(f"Unknown text encoder: {text_type}. Chọn: distilbert, bert")

    return img_enc, txt_enc, tokenizer_name


def get_checkpoint_name(image_type, text_type):
    """Tạo tên checkpoint dựa trên config."""
    return f"best_{image_type}_{text_type}.pth"
