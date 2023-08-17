import logging
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from rich.logging import RichHandler
from transformers import T5Tokenizer, T5TokenizerFast
from wonderwords import RandomWord

from src.datamodule import DocT5QueryDataModule
from src.module import DocT5QueryModule

torch.set_float32_matmul_precision("medium")


def _parse_args() -> Namespace:
    parser = ArgumentParser()

    paths = parser.add_argument_group("paths", "Paths to data and model")
    # Multiple paths accepted
    paths.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path to dataset directory",
    )
    paths.add_argument("--model_path", type=str, required=True, help="Model path")
    paths.add_argument("--tokenizer_path", type=str, default=None, help="Path to vocab file")

    seeds = parser.add_argument_group("seeds", "Seeds for reproducibility")
    seeds.add_argument("--seed", type=int, default=42, help="Seed for everything")

    trainer = parser.add_argument_group("trainer", "Trainer arguments")
    trainer.add_argument("--batch_size", type=int, default=16, help="Batch size")
    trainer.add_argument("--max_steps", type=int, default=-1, help="Max number of steps")
    trainer.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer",
    )
    trainer.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    trainer.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    trainer.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        help="Accelerator for training (cpu, cuda, ...)",
    )
    trainer.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Strategy for training (auto, ddp, ...)",
    )
    trainer.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to train on (1 for single GPU)",
    )
    trainer.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Floating point precision (32, 16, bf16)",
    )
    trainer.add_argument(
        "--val_check_interval",
        type=eval,
        default=1.0,
        help="Validation check interval. If int, check every n steps. If float, check every n percent of each epoch.",
    )
    trainer.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients on before performing optimizer step",
    )

    logger = parser.add_argument_group("logger", "Logger arguments")
    logger.add_argument("--project_name", type=str, default="kodocT5query", help="Project name")
    logger.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    logger.add_argument("--wandb_entity", type=str, default="kocohub", help="WandB entity name")
    logger.add_argument("--wandb_tags", type=str, nargs="+", default=None, help="WandB tags")

    parsed = parser.parse_args()

    # Check arguments
    if parsed.tokenizer_path is None:
        parsed.tokenizer_path = parsed.model_path

    return parsed


# noinspection PyUnusedLocal
def train(
    dataset_paths: list[str],
    model_path: str,
    tokenizer_path: str,
    seed: int,
    accelerator: Literal["cpu", "cuda"],
    strategy: None | Literal["ddp"],
    batch_size: int,
    num_workers: int,
    max_steps: int,
    devices: int,
    lr: float,
    val_check_interval: int | float,
    accumulate_grad_batches: int,
    use_fast_tokenizer: bool = False,
    project_name: str | None = "kodocT5query",
    use_wandb: bool = False,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
    python_logger: logging.Logger | None = None,
    *args,
    **kwargs,
):
    """
    docT5query trainer

    Args:
        dataset_paths (list[str]): List of paths to dataset directories
        model_path (str): Model path
        tokenizer_path (str): Path to tokenizer
        seed (int): Random seed
        accelerator (str): Accelerator for training (cpu, cuda)
        strategy (str): Strategy for training
        batch_size (int): Batch size
        num_workers (int): Number of workers for dataloader
        max_steps (int): Max number of steps
        devices (int): Number of devices
        lr (float): Learning rate
        val_check_interval (int | float): Validation check interval. If int, check every n steps. If float,
            check every n percent of each epoch
        accumulate_grad_batches (int): Accumulate gradient batches
        use_fast_tokenizer (bool): Whether to use fast tokenizer
        project_name (str): WandB project name
        use_wandb (bool): Whether to use wandb
        wandb_entity (str): WandB entity name
        wandb_tags (list[str]): WandB tags
        python_logger (logging.Logger): Logger to use. If None, use default logger
        *args: Additional args
        **kwargs: Additional kwargs

    Returns:
        None

    """
    if python_logger is None:
        python_logger = logging.getLogger(__name__)

    python_logger.info("Starting training")
    seed_everything(seed)

    if use_fast_tokenizer:
        python_logger.info("Using fast tokenizer")
        tokenizer_cls = T5TokenizerFast
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        python_logger.info("Using Python tokenizer")
        tokenizer_cls = T5Tokenizer

    # Create a unique save path
    random_word_generator = RandomWord()
    while True:
        random_word = random_word_generator.random_words(include_parts_of_speech=["nouns"])[0]
        save_pretrained_path = Path("output") / random_word

        if not save_pretrained_path.exists():
            break

    save_pretrained_path.mkdir(parents=True)
    python_logger.info(f"Model will be saved to {save_pretrained_path.absolute()}")

    tokenizer = tokenizer_cls.from_pretrained(tokenizer_path)
    python_logger.info(f"Using tokenizer {tokenizer_cls.__name__}")

    # Load model and datamodule
    python_logger.info(f"Loading model from {model_path}")
    module = DocT5QueryModule(
        model_path=model_path,
        lr=lr,
    )
    python_logger.info(f"Loading {len(dataset_paths)} datasets from {dataset_paths}")
    datamodule = DocT5QueryDataModule(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Setup trainer
    callbacks = [
        ModelCheckpoint(
            dirpath=save_pretrained_path,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="{epoch}-{val_loss:.2f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
        ),
        RichModelSummary(max_depth=2),
        RichProgressBar(),
    ]

    loggers = [CSVLogger(save_dir="logs", name=project_name)]

    if use_wandb:
        python_logger.info("Using wandb")
        loggers.append(
            WandbLogger(
                project=project_name,
                entity=wandb_entity,
                name=f"{random_word}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=wandb_tags,
            ),
        )

    trainer = Trainer(
        enable_model_summary=False,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        # Step-based training
        max_epochs=-1,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        # Gradient accumulation
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
    )

    for pl_logger in loggers:
        pl_logger.log_hyperparams(
            {
                "dataset_paths": dataset_paths,
                "model_path": model_path,
                "tokenizer_path": tokenizer_path,
                "save_pretrained_path": save_pretrained_path.absolute(),
                "accelerator": accelerator,
                "strategy": strategy,
                "devices": devices,
                "seed": seed,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_steps": max_steps,
                "lr": lr,
                "val_check_interval": val_check_interval,
                "accumulate_grad_batches": accumulate_grad_batches,
                "use_fast_tokenizer": use_fast_tokenizer,
            },
        )

    python_logger.info("Training started")
    trainer.fit(module, datamodule=datamodule)

    python_logger.info("Testing started")
    trainer.test(ckpt_path="best", datamodule=datamodule)

    # noinspection PyUnresolvedReferences
    python_logger.info(
        f"Training finished.\n"
        f"Best path: {trainer.checkpoint_callback.best_model_path}\n"
        f"Best score: {trainer.checkpoint_callback.best_model_score}\n"
    )

    # Save model
    python_logger.info(f"Saving model to {save_pretrained_path}")

    module.model.save_pretrained(save_pretrained_path)
    tokenizer.save_pretrained(save_pretrained_path)


def _main():
    python_logger = logging.getLogger("Trainer")
    python_logger.setLevel(logging.INFO)
    python_logger.addHandler(RichHandler())

    args = _parse_args()
    train(python_logger=python_logger, **vars(args))


if __name__ == "__main__":
    _main()
