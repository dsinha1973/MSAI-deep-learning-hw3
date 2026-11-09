import argparse

import torch
import torch.nn as nn

from homework.models import Detector, save_model
from homework.metrics import DetectionMetric
from homework.datasets.road_dataset import load_data


def evaluate(model, loader, device):
    model.eval()

    metric = DetectionMetric()

    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()

    total_loss = 0.0

    with torch.inference_mode():

        for batch in loader:

            images = batch["image"].to(device)
            tracks = batch["track"].to(device)
            depths = batch["depth"].to(device)

            logits, pred_depth = model(images)

            seg_loss = seg_criterion(
                logits,
                tracks,
            )

            dep_loss = depth_criterion(
                pred_depth,
                depths,
            )

            loss = seg_loss + dep_loss

            total_loss += loss.item()

            preds = logits.argmax(dim=1)

            metric.add(
                preds,
                tracks,
                pred_depth,
                depths,
            )

    metrics = metric.compute()

    return (
        total_loss / len(loader),
        metrics,
    )


def train(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader = load_data(
        args.train_dir,
        batch_size=args.batch_size,
        shuffle=True,
        transform_pipeline="default",
    )

    val_loader = load_data(
        args.val_dir,
        batch_size=args.batch_size,
        shuffle=False,
        transform_pipeline="default",
    )

    model = Detector().to(device)

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    seg_criterion = nn.CrossEntropyLoss()

    depth_criterion = nn.L1Loss()

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

    best_iou = 0.0

    for epoch in range(args.epochs):

        model.train()

        detector_metric = DetectionMetric()

        running_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(device)
            tracks = batch["track"].to(device)
            depths = batch["depth"].to(device)

            optimizer.zero_grad()

            logits, pred_depth = model(images)

            seg_loss = seg_criterion(
                logits,
                tracks,
            )

            dep_loss = depth_criterion(
                pred_depth,
                depths,
            )

            loss = seg_loss + dep_loss

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            preds = logits.argmax(dim=1)

            detector_metric.add(
                preds,
                tracks,
                pred_depth,
                depths,
            )

        scheduler.step()

        train_metrics = detector_metric.compute()

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch+1:02d}/{args.epochs}"
        )

        print(
            f"Train Loss: {running_loss/len(train_loader):.4f}"
        )

        print(
            f"Train IoU: {train_metrics['iou']:.4f}"
        )

        print(
            f"Train Acc: {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        print(
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Depth Error: "
            f"{val_metrics['abs_depth_error']:.4f}"
        )

        print(
            f"Val TP Depth Error: "
            f"{val_metrics['tp_depth_error']:.4f}"
        )

        print("-" * 50)

        if val_metrics["iou"] > best_iou:

            best_iou = val_metrics["iou"]

            path = save_model(model)

            print(
                f"Saved best model "
                f"(IoU={best_iou:.4f}) "
                f"to {path}"
            )

    print()
    print("Training Complete")
    print(f"Best IoU: {best_iou:.4f}")


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        type=str,
        default="drive_data/train",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default="drive_data/val",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(get_args())