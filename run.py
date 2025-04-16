from transformer import MTLModel
from trainer import MTLTrainer

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

dataset = "Sp1786/multiclass-sentiment-analysis-dataset"
model = MTLModel()
# model.get_data(dataset)
for i, emb in enumerate(embeddings):
    min_val = emb.min().item()
    max_val = emb.max().item()
    mean_val = emb.mean().item()
    std_val = emb.std().item()
    median_val = emb.median().item()

    print(f"Embedding {i+1}:")
    print(f"  Min:    {min_val:.4f}")
    print(f"  Max:    {max_val:.4f}")
    print(f"  Mean:   {mean_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  Std:    {std_val:.4f}")
    print(f"  Values: {emb.tolist()}\n")
