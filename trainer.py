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
        # assume s
        data = []
        for sample in sentences:
            text = sample["text"]
            sentiment_label = SENTIMENT_MAP[sample["sentiment"]] if sample.get("sentiment") in SENTIMENT_MAP else None
            class_label = CLASS_MAP[sample["class"]] if sample.get("class") in CLASS_MAP else None
            data.append((text, sentiment_label, class_label))

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                texts, sentiment_labels, class_labels = zip(*batch)

                # Tokenize
                encoded = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                # Convert labels to tensors (masking None with -100, ignored by CrossEntropyLoss)
                sentiment_targets = torch.tensor([
                    label if label is not None else -100 for label in sentiment_labels
                ], dtype=torch.long).to(self.device)

                class_targets = torch.tensor([
                    label if label is not None else -100 for label in class_labels
                ], dtype=torch.long).to(self.device)

                # Forward pass
                topic_logits, sentiment_logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Compute loss only on valid labels
                loss = 0
                if (class_targets != -100).any():
                    class_loss = self.loss_fn(topic_logits, class_targets)
                    loss += class_loss

                if (sentiment_targets != -100).any():
                    sentiment_loss = self.loss_fn(sentiment_logits, sentiment_targets)
                    loss += sentiment_loss

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(dataloader):.4f}")

