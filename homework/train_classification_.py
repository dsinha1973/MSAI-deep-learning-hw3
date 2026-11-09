import argparse
from pathlib import Path

import torch
import torch.nn as nn

from homework.models import Classifier, save_model
from homework.metrics import AccuracyMetric
from homework.datasets.classification_dataset import load_data


def evaluate(model, loader, device):
    model.eval()

    metric = AccuracyMetric()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            metric.add(preds, labels)

    metrics = metric.compute()

    return (
        total_loss / len(loader),
        metrics["accuracy"],
    )


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader = load_data(
        dataset_path=args.train_dir,
        transform_pipeline="aug",
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = load_data(
        dataset_path=args.val_dir,
        transform_pipeline="default",
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = Classifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5,
    )

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0

        metric = AccuracyMetric()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            preds = logits.argmax(dim=1)
            metric.add(preds, labels)

        scheduler.step()

        train_acc = metric.compute()["accuracy"]

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc

            model_path = save_model(model)

            print(
                f"Saved best model "
                f"(acc={best_acc:.4f}) -> {model_path}"
            )

    print(f"Best Validation Accuracy: {best_acc:.4f}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        type=str,
        default="classification_data/train",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default="classification_data/val",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())