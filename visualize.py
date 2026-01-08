import argparse
from pathlib import Path

import fiftyone as fo
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AdvancedAlexNet


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

IMAGE_PATH = Path.cwd() / "data" / "cifar10_images"
IMAGE_PATH.mkdir(parents=True, exist_ok=True)


def make_cifar10_data_loader(batch_size: int = 32) -> DataLoader:
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def predict(model: torch.nn.Module, imgs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = model(imgs).detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)
    odds = np.exp(logits)
    confidences = np.max(odds, axis=1) / np.sum(odds, axis=1)
    return predictions, confidences, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the trained model file", default="best_model.pth")
    args = parser.parse_args()

    # Load model
    model = AdvancedAlexNet()
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_loader = make_cifar10_data_loader()

    samples = []
    for imgs, targets in tqdm(data_loader, desc="Predicting"):
        # Make predictions
        predictions, probabilities, logits_ = predict(model, imgs.to(device))
        for image, prediction, probability, target in zip(imgs, predictions, probabilities, targets):
            # Denormalize and save image
            torchvision.utils.save_image(
                image * torch.tensor(CIFAR10_STD).reshape(3, 1, 1) + torch.tensor(CIFAR10_MEAN).reshape(3, 1, 1),
                IMAGE_PATH / f"{len(samples)}.png",
            )

            # Create FiftyOne sample
            sample = fo.Sample(
                filepath=str(IMAGE_PATH / f"{len(samples)}.png"),
                ground_truth=fo.Classification(label=CLASSES[target]),
                prediction=fo.Classification(label=CLASSES[prediction], confidence=float(probability)),
            )

            # Tag misclassified samples
            if prediction != target:
                sample["prediction"].tags.append("misclassified")
                sample.tags.append("misclassified")
            samples.append(sample)

    # Create FiftyOne dataset and launch app
    dataset = fo.Dataset(name="cifar10_predictions", overwrite=True)
    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    session.wait()  # Keep the app running


if __name__ == "__main__":
    main()
