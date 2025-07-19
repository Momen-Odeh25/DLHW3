import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

from .datasets.road_dataset import load_data
from .models import Detector
from .metrics import DetectionMetric

def train(
    exp_dir: str = "logs_det",
    model_name: str = "detector",
    num_epoch: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
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

    model = Detector(**kwargs).to(device)
    model.train()

    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="default",
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = load_data(
        "drive_data/val",
        transform_pipeline="default",
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()
    metric = DetectionMetric()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(1, num_epoch + 1):
        model.train()
        for batch in train_loader:
            imgs     = batch["image"].to(device)
            seg_gt   = batch["track"].to(device)
            depth_gt = batch["depth"].to(device)
            optimizer.zero_grad()
            logits, raw_depth = model(imgs)
            loss_seg = seg_loss_fn(logits, seg_gt)
            loss_depth = depth_loss_fn(raw_depth, depth_gt)
            loss = loss_seg + loss_depth
            loss.backward()
            optimizer.step()

            logger.add_scalar("train_seg_loss", loss_seg.item(), global_step)
            logger.add_scalar("train_depth_loss", loss_depth.item(), global_step)
            global_step += 1

        model.eval()
        metric.reset()
        with torch.inference_mode():
            for batch in val_loader:
                imgs     = batch["image"].to(device)
                seg_gt   = batch["track"].to(device)
                depth_gt = batch["depth"].to(device)
                seg_pred, depth_pred = model.predict(imgs)
                metric.add(seg_pred, seg_gt, depth_pred, depth_gt)

        det_metrics = metric.compute()
        miou = det_metrics["iou"]
        mae = det_metrics["abs_depth_error"]
        lane_mae = det_metrics["tp_depth_error"]

        logger.add_scalar("val_mean_iou", miou, epoch)
        logger.add_scalar("val_depth_mae", mae, epoch)
        logger.add_scalar("val_depth_mae_on_lane", lane_mae, epoch)

        print(f"Epoch {epoch:02d}/{num_epoch:02d}: "
              f"mean_iou={miou:.4f} depth_mae={mae:.4f} lane_mae={lane_mae:.4f}")

    torch.save(model.state_dict(), f"{model_name}.pth")
    torch.save(model.state_dict(), log_dir / f"{model_name}.pth")
    print(f"Model saved to {model_name}.pth and checkpoint to {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir",    type=str, default="logs_det")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch",  type=int, default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed",       type=int, default=2024)
    args = parser.parse_args()
    train(**vars(args))
