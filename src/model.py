import torch
import lightning.pytorch as pl
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import BCEWithLogitsLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)


class FinanciaMultilabel(pl.LightningModule):
    def __init__(self, model, num_labels):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.loss = BCEWithLogitsLoss()
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask, token_type_ids).logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch["token_type_ids"]
        outputs = self(input_ids, attention_mask, token_type_ids)
        loss = self.loss(
            outputs.view(-1, self.num_labels),
            labels.type_as(outputs).view(-1, self.num_labels),
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch["token_type_ids"]
        outputs = self(input_ids, attention_mask, token_type_ids)
        loss = self.loss(
            outputs.view(-1, self.num_labels),
            labels.type_as(outputs).view(-1, self.num_labels),
        )
        pred_labels = torch.sigmoid(outputs)
        info = {"val_loss": loss, "pred_labels": pred_labels, "labels": labels}
        self.validation_step_outputs.append(info)
        return

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        pred_labels = torch.cat([x["pred_labels"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        threshold = 0.50
        pred_bools = pred_labels > threshold
        true_bools = labels == 1
        val_f1_accuracy = (
            f1_score(true_bools.cpu(), pred_bools.cpu(), average="micro") * 100
        )
        val_flat_accuracy = accuracy_score(true_bools.cpu(), pred_bools.cpu()) * 100
        self.log("val_loss", avg_loss)
        self.log("val_f1_accuracy", val_f1_accuracy, prog_bar=True)
        self.log("val_flat_accuracy", val_flat_accuracy, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=2, verbose=True, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def load_model(checkpoint_path, model, num_labels, device):
    model_hugginface = AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model = FinanciaMultilabel.load_from_checkpoint(
        checkpoint_path,
        model=model_hugginface,
        num_labels=num_labels,
        map_location=device,
    )
    return model
