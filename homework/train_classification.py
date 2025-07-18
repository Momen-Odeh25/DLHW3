# File: homework/train_classification.py

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .datasets.classification_dataset import load_data
from .models import Classifier
from .metrics import AccuracyMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = Classifier(**kwargs).to(device)
    model.train()

    train_loader = load_data(
        "classification_data/train",
        transform_pipeline="aug",
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = load_data(
        "classification_data/val",
        transform_pipeline="default",
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metric = AccuracyMetric()

    global_step = 0
    for epoch in range(num_epoch):
        # --- training ---
        model.train()
        train_accs = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_accs.append((preds == labels).float().mean().item())

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        model.eval()
        metric.reset()
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model.predict(imgs)
                metric.add(preds, labels)
        
        val_metrics = metric.compute()
        val_acc = val_metrics["accuracy"]

        train_epoch_acc = float(np.mean(train_accs))
        logger.add_scalar("train_accuracy", train_epoch_acc, epoch)
        logger.add_scalar("val_accuracy", val_acc, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:02d}/{num_epoch:02d}: "
                f"train_acc={train_epoch_acc:.4f} val_acc={val_acc:.4f}"
            )

    torch.save(model.state_dict(), f"{model_name}.pth")
    torch.save(model.state_dict(), log_dir / f"{model_name}.pth")
    print(f"Model saved to {model_name}.pth and logs to {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="classifier")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    train(**vars(args))
