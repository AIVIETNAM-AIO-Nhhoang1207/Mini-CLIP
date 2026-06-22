import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertConfig

class ImageEncoder(nn.Module):
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

