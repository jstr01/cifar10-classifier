import datetime
import logging
import os
import time

import torch
import torch.utils.data
from utils import AverageMeter
from torch import nn

import argparse
from dataloader import get_dataloaders
from model import build_model
from utils import seed_everything

import mlflow

logging.basicConfig(level=logging.INFO)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()

    for i, (image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)

        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        accuracy_meter.update(
            (output.argmax(dim=1) == target).float().mean().item(), n=image.size(0)
        )
        loss_meter.update(loss.item(), n=image.size(0))

        if i % 50 == 0:
            logging.info(
                f"Epoch [{epoch}] Iter [{i}/{len(data_loader)}] Loss: {loss_meter.avg:.4f} Acc: {accuracy_meter.avg:.4f}"
            )

    return loss_meter.avg, accuracy_meter.avg


def evaluate(model, criterion, data_loader, device):
    model.eval()
    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)

            accuracy_meter.update(
                (output.argmax(dim=1) == target).float().mean().item(), n=image.size(0)
            )
            loss_meter.update(loss.item(), n=image.size(0))

    return loss_meter.avg, accuracy_meter.avg


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs
):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch
        )
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)

        scheduler.step()

        # if improved validation accuracy, save & log the model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # save best model locally and log as artifact
            best_model_path = os.path.join(os.getcwd(), "best_model.pth")
            torch.save(model.state_dict(), best_model_path)

            mlflow.log_artifact(best_model_path, artifact_path="models")

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

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="alexnet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-url", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.mlflow_url:
        mlflow.set_tracking_uri(args.mlflow_url)
        logging.info(f"MLflow tracking URI set to: {args.mlflow_url}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size, target_size=227 if args.model == "alexnet" else 32
    )

    model = build_model(name=args.model, num_classes=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
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
            device,
            args.epochs,
        )

        # save final model artifact
        final_model_path = os.path.join(os.getcwd(), "final_model.pth")
        torch.save(trained_model.state_dict(), final_model_path)

        mlflow.log_artifact(final_model_path, artifact_path="models")


if __name__ == "__main__":
    main()
