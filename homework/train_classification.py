import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.classification_dataset import load_data
from models import Classifier
from metrics import AccuracyMetric

#Hyperparameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Data loaders using provided load_data function
train_loader = load_data(
    dataset_path='classification_data',
    transform_pipeline='aug',
    return_dataloader=True,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = load_data(
    dataset_path='classification_data',
    transform_pipeline='default',
    return_dataloader=True,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = Classifier().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
metric = AccuracyMetric()

best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    metric.reset()

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model.predict(imgs)
            metric.add(preds, labels)

    val_acc = metric.value()
    print(f"Epoch {epoch:02d}: val accuracy = {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_classifier.pth')
        print(f"New best model (acc={best_acc:.4f}) saved.")

    if val_acc >= 0.80:
        print("Reached target accuracy ≥ 0.80, stopping training.")
        break