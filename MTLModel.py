from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

# 0: negative, 1: neutral, 2: positive
# Sentence Classification by topic: Politics, sports, technology, health, entertainment
# Sentiment Analysis: 0: negative, 1: neutral, 2: positive

SENTIMENT_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
CLASS_MAP = {'politics': 0, 'sports': 1, 'technology': 2, 'health': 3, 'entertainment': 4}
class MTLModel(nn.Module):
    def __init__(self):
        super(MTLModel, self).__init__()
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.hidden_size = self.encoder.config.hidden_size
        
        self.topic_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

        self.sentiment_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        model_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(model_output, attention_mask)
        topic_logits = self.topic_head(pooled)
        sentiment_logits = self.sentiment_head(pooled)
        return topic_logits, sentiment_logits