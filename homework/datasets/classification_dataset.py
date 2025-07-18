import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from classification_dataset import load_data
from models import Classifier


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

    # load model
    model = Classifier(**kwargs).to(device)
    model.train()

    # data loaders
    train_loader = load_data(
        dataset_path='classification_data/train',
        transform_pipeline='aug',
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = load_data(
        dataset_path='classification_data/val',
        transform_pipeline='default',
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(num_epoch):
        # training epoch
        model.train()
        train_accs = []
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == label).float().mean().item()
            train_accs.append(acc)

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # validation epoch
        model.eval()
        val_accs = []
        with torch.inference_mode():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == label).float().mean().item()
                val_accs.append(acc)

        # log metrics
        train_epoch_acc = float(np.mean(train_accs))
        val_epoch_acc = float(np.mean(val_accs))
        logger.add_scalar("train_accuracy", train_epoch_acc, epoch)
        logger.add_scalar("val_accuracy", val_epoch_acc, epoch)

        # print progress
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{num_epoch:02d}: train_acc={train_epoch_acc:.4f} val_acc={val_epoch_acc:.4f}")

    # save final model
    torch.save(model.state_dict(), f"{model_name}.pth")
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