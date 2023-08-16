from pytorch_lightning import LightningModule
from torch.optim import AdamW

from transformers import T5ForConditionalGeneration


class DocT5QueryModule(LightningModule):
    def __init__(self, model_path: str, lr: float = 1e-4):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.lr = lr

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        return optimizer

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        del batch["labels"]

        outputs = self.model.generate(**batch)

        return outputs
