import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

SENTIMENT_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
CLASS_MAP = {'politics': 0, 'sports': 1, 'technology': 2, 'health': 3, 'entertainment': 4}

class MTLTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, sentences, batch_size=8, epochs=3):
        self.model.train()
        num_samples = len(sentences)
        embeddings = torch.tensor([sample['embedding'] for sample in sentences], dtype=torch.float32).to(self.device)
        class_labels = torch.tensor([sample['class'] if sample['class'] is not None else -100 for sample in sentences], dtype=torch.long).to(self.device)
        sentiment_labels = torch.tensor([sample['sentiment'] if sample['sentiment'] is not None else -100 for sample in sentences], dtype=torch.long).to(self.device)
        for epoch in range(epochs):
            total_loss = 0
            for start in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}"):
                end = start + batch_size
                batch_emb = embeddings[start:end]
                batch_class_labels = class_labels[start:end]
                batch_sentiment_labels = sentiment_labels[start:end]

                # Forward through classifier heads (embedding already computed)
                topic_logits = self.model.topic_head(batch_emb)
                sentiment_logits = self.model.sentiment_head(batch_emb)

                # Compute masked losses
                loss = 0
                if (batch_class_labels != -100).any():
                    class_loss = self.loss_fn(topic_logits, batch_class_labels)
                    loss += class_loss

                if (batch_sentiment_labels != -100).any():
                    sentiment_loss = self.loss_fn(sentiment_logits, batch_sentiment_labels)
                    loss += sentiment_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} - Avg Loss: {total_loss / (num_samples // batch_size):.4f}")