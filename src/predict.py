from pathlib import Path
import torch
import numpy as np

from .tokenizer import load_tokenizer, preprocessing_text
from .model import load_model

# CONFIG
NUM_VARAIBLES = 3
NUM_LABELS = 3
num_labels = NUM_LABELS * NUM_VARAIBLES
divice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = NUM_LABELS * NUM_VARAIBLES
model_name = "pysentimiento/robertuito-sentiment-analysis"

checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "model.ckpt"
tokenizer = load_tokenizer(model_name)
model = load_model(checkpoint_path, model_name, num_labels, divice)


RETURN_VALUES = [
    "target_sentiment_negative",
    "target_sentiment_neutral",
    "target_sentiment_positive",
    "companies_sentiment_negative",
    "companies_sentiment_neutral",
    "companies_sentiment_positive",
    "consumers_sentiment_negative",
    "consumers_sentiment_neutral",
    "consumers_sentiment_positive",
]


def filter(preds, threshold=0.5):
    bool = preds > threshold
    indices = np.where(bool)[0]
    filtered_values = {RETURN_VALUES[index]: preds[index] for index in indices}
    return filtered_values


def get_predict(text):
    inputs = preprocessing_text(text, tokenizer)
    input_ids = inputs["input_ids"].to(divice)
    attention_mask = inputs["attention_mask"].to(divice)
    token_type_ids = inputs["token_type_ids"].to(divice)
    outputs = model(input_ids, attention_mask, token_type_ids)
    preds = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = filter(preds[0])
    return preds
