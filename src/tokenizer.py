from transformers import (
    AutoTokenizer,
)


def load_tokenizer(model_tokenizer):
    """Load the tokenizer"""
    return AutoTokenizer.from_pretrained(model_tokenizer)


def preprocessing_text(text, tokenizer):
    """Tokenize the text"""
    return tokenizer.encode_plus(
        text,
        max_length=130,
        pad_to_max_length=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
