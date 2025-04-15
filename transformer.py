from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
import torch.nn as nn

# 0: negative, 1: neutral, 2: positive
# Sentence Classification by topic: Politics, sports, technology, health, entertainment
# Sentiment Analysis: 0: negative, 1: neutral, 2: positive

class MTLModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.hidden_size = self.model.config.hidden_size
        self.topic_head = nn.Linear(self.hidden_size, 5) # politics, sports, technology, health, entertainment
        self.sentiment_head = nn.Linear(self.hidden_size, 3) # negative, neutral, positive
        
    # def get_data(self, dataset):
    #     ds = load_dataset(dataset) # Data Format: assume there are labels "sentiment" and "class"
    #     # Sentiment Values: negative, neutral, positive
    #     # Class Values: politics, sports, technology, health, entertainment
    #     self.ds = ds.train_test_split(test_size=0.2, shuffle=True)

    def get_embeddings(self, sentences):
        sentence_text = [sentence['text'] for sentence in sentences]
        encoded_input = self.tokenizer(sentence_text, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input) 
            print("Shape using index:", model_output[0].shape)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings        
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)