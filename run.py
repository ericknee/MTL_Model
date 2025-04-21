from MTLModel import MTLModel
from MTLTrainer import MTLTrainer
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Mappings (reuse from above)
SENTIMENT_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
CLASS_MAP = {'politics': 0, 'sports': 1, 'technology': 2, 'health': 3, 'entertainment': 4}

# Sentences dataset (your provided data)
sentences = [
    {"text": "The government's response to the crisis was utterly disappointing.", "class": "politics", "sentiment": "negative"},
    {"text": "The team’s comeback victory was incredibly thrilling to watch.", "class": "sports", "sentiment": "positive"},
    {"text": "The new smartphone was released with standard features this year.", "class": "technology", "sentiment": "neutral"},
    {"text": "The side effects of the new medication were concerning.", "class": "health", "sentiment": "negative"},
    {"text": "That movie was a masterpiece from start to finish.", "class": "entertainment", "sentiment": "positive"},
    {"text": "The senator introduced a new bill in the house today.", "class": "politics", "sentiment": "neutral"},
    {"text": "The referee's decisions ruined the entire match experience.", "class": "sports", "sentiment": "negative"},
    {"text": "This app makes managing tasks so much easier and faster.", "class": "technology", "sentiment": "positive"},
    {"text": "The doctor explained the procedure clearly and professionally.", "class": "health", "sentiment": "neutral"},
    {"text": "The show’s plot was predictable and lacked emotional depth.", "class": "entertainment", "sentiment": "negative"}
]

for sample in sentences:
    sample['class'] = CLASS_MAP.get(sample['class'], None)
    sample['sentiment'] = SENTIMENT_MAP.get(sample['sentiment'], None)

model = MTLModel()
trainer = MTLTrainer(model)
trainer.train(sentences, batch_size=4, epochs=3)
