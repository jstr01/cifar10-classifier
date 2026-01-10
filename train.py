import argparse
import datetime
import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn

from dataloader import get_dataloaders
from model import AdvancedAlexNet, MiniAlexNet
from utils import AverageMeter, seed_everything

logging.basicConfig(level=logging.INFO)

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


def train_one_epoch(model, criterion, optimizer, data_loader, epoch):
    model.train()
    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()

    for i, (image, target) in enumerate(data_loader):
        image, target = image.to(DEVICE), target.to(DEVICE)

        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        accuracy_meter.update((output.argmax(dim=1) == target).float().mean().item(), n=image.size(0))
        loss_meter.update(loss.item(), n=image.size(0))

        if i % 50 == 0:
            logging.info(
                f"Epoch [{epoch}] Iter [{i}/{len(data_loader)}] Loss: {loss_meter.avg:.4f} Acc: {accuracy_meter.avg:.4f}"
            )

    return loss_meter.avg, accuracy_meter.avg


def evaluate(model, criterion, data_loader):
    model.eval()
    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(DEVICE), target.to(DEVICE)

            output = model(image)
            loss = criterion(output, target)

            accuracy_meter.update((output.argmax(dim=1) == target).float().mean().item(), n=image.size(0))
            loss_meter.update(loss.item(), n=image.size(0))

    return loss_meter.avg, accuracy_meter.avg


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_accuracy = 0.0
    best_model_path = Path.cwd() / "best_model.pth"
    last_model_path = Path.cwd() / "final_model.pth"

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, epoch)
        val_loss, val_acc = evaluate(model, criterion, val_loader)

        scheduler.step()

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)

        epoch_time = time.time() - start_time
        logging.info(
            f"Epoch [{epoch}] completed in {str(datetime.timedelta(seconds=int(epoch_time)))}"
            f" - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Best Val Acc: {best_accuracy:.4f}"
        )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("best_val_acc", best_accuracy, step=epoch)
        mlflow.log_metric("epoch_time_sec", epoch_time, step=epoch)

    torch.save(model.state_dict(), last_model_path)

    # log models to mlflow
    mlflow.log_artifact(str(best_model_path), artifact_path="models")
    mlflow.log_artifact(str(last_model_path), artifact_path="models")

    # return best model
    model.load_state_dict(torch.load(best_model_path))
    return model


def final_report(model, val_loader):
    """confusion matrix and classification report"""

    # Make predictions on test set
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            prediction = outputs.argmax(dim=1)
            predictions.extend(prediction.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Create and log confusion matrix
    cm = confusion_matrix(labels, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    cm_display.plot(cmap="Blues")
    cm_figure_path = Path.cwd() / "confusion_matrix.png"
    cm_display.figure_.savefig(cm_figure_path)
    mlflow.log_artifact(str(cm_figure_path), artifact_path="reports")

    # Create and log classification report
    cr = classification_report(labels, predictions, target_names=CLASSES)
    cr_figure_path = Path.cwd() / "classification_report.txt"
    with open(cr_figure_path, "w") as f:
        f.write(cr)
    mlflow.log_artifact(str(cr_figure_path), artifact_path="reports")

    # Log to console
    logging.info("Confusion Matrix:\n" + str(cm))
    logging.info("Classification Report:\n" + str(cr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="advancedalexnet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-url", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.mlflow_url:
        mlflow.set_tracking_uri(args.mlflow_url)
        logging.info(f"MLflow tracking URI set to: {args.mlflow_url}")

    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)

    # Initialize model
    model = None
    match args.model:
        case "minialexnet":
            model = MiniAlexNet(num_classes=10)
        case "advancedalexnet":
            model = AdvancedAlexNet(num_classes=10)
        case _:
            raise ValueError(f"Unknown model name: {args.model}")
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mlflow.set_experiment("cifar10_classification")

    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("model", args.model)
        mlflow.log_param("seed", args.seed)

        trained_model = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            args.epochs,
        )

        final_report(trained_model, val_loader)


if __name__ == "__main__":
    main()
