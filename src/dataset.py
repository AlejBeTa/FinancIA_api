from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)
import torch


class FinanciaSentimental(Dataset):
    """This class is used to load the data and tokenize it"""

    def __init__(self, tokenizer, dataframe, columns, max_len=512):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        ## Columns to target
        self._columns = columns
        self.max_len = max_len

    @property
    def columns(self):
        """Return the columns to target"""
        return self._columns

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Get the data at the index"""
        values = self.dataframe.iloc[index]
        text = values["text"]
        label = values[self._columns].values.astype(np.float32)
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=130,
            pad_to_max_length=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = torch.tensor(label, dtype=torch.float)
        input_ids = inputs["input_ids"].squeeze().to(dtype=torch.long)
        attention_mask = inputs["attention_mask"].squeeze().to(dtype=torch.long)
        token_type_ids = inputs["token_type_ids"].squeeze().to(dtype=torch.long)

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": label,
        }

        return inputs_dict
