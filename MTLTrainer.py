import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


SENTIMENT_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
CLASS_MAP = {'politics': 0, 'sports': 1, 'technology': 2, 'health': 3, 'entertainment': 4}

class MTLTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, sentences, batch_size=16, epochs=3):
        self.model.train()
        num_samples = len(sentences)

        texts = [sample['text'] for sample in sentences]
        class_labels = torch.tensor(
            [sample['class'] if sample['class'] is not None else -100 for sample in sentences], dtype=torch.long).to(self.device)
        sentiment_labels = torch.tensor(
            [sample['sentiment'] if sample['sentiment'] is not None else -100 for sample in sentences], dtype=torch.long).to(self.device)

        for epoch in range(epochs):
            total_loss = 0
            all_topic_preds, all_topic_targets = [], []
            all_sentiment_preds, all_sentiment_targets = [], []

            for start in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}"):
                end = start + batch_size
                batch_texts = texts[start:end]
                batch_class_labels = class_labels[start:end]
                batch_sentiment_labels = sentiment_labels[start:end]

                # Tokenize
                encoded = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                # Model forward()
                topic_logits, sentiment_logits = self.model(input_ids, attention_mask)

                # Masked loss
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

                # Predictions
                topic_preds = topic_logits.argmax(dim=1)
                sentiment_preds = sentiment_logits.argmax(dim=1)
                class_mask = (batch_class_labels != -100)
                sentiment_mask = (batch_sentiment_labels != -100)

                if class_mask.any():
                    all_topic_preds.extend(topic_preds[class_mask].cpu().tolist())
                    all_topic_targets.extend(batch_class_labels[class_mask].cpu().tolist())

                if sentiment_mask.any():
                    all_sentiment_preds.extend(sentiment_preds[sentiment_mask].cpu().tolist())
                    all_sentiment_targets.extend(batch_sentiment_labels[sentiment_mask].cpu().tolist())

            # Metrics
            topic_acc = accuracy_score(all_topic_targets, all_topic_preds) if all_topic_targets else 0
            topic_f1 = f1_score(all_topic_targets, all_topic_preds, average='macro') if all_topic_targets else 0
            sentiment_acc = accuracy_score(all_sentiment_targets, all_sentiment_preds) if all_sentiment_targets else 0
            sentiment_f1 = f1_score(all_sentiment_targets, all_sentiment_preds, average='macro') if all_sentiment_targets else 0

            avg_loss = total_loss / (num_samples // batch_size)
            print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            print(f"  Topic     → Accuracy: {topic_acc:.4f}, F1: {topic_f1:.4f}")
            print(f"  Sentiment → Accuracy: {sentiment_acc:.4f}, F1: {sentiment_f1:.4f}\n")