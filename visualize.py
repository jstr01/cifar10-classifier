import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
import torchvision
from fiftyone.utils.torch import TorchImageDataset
from torch.utils.data import DataLoader

from model import AdvancedAlexNet


def make_cifar10_data_loader(image_paths: list[str], sample_ids: list[int], batch_size: int = 32) -> DataLoader:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )
    dataset = TorchImageDataset(image_paths, sample_ids=sample_ids, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)


def predict(model: torch.nn.Module, imgs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = model(imgs).detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)
    odds = np.exp(logits)
    confidences = np.max(odds, axis=1) / np.sum(odds, axis=1)
    return predictions, confidences, logits


def main():
    # Load model
    model = AdvancedAlexNet()
    state_dict = torch.load(
        "mlflow_data/mlartifacts/1/c472357933a34ec780d4a5312d482d06/artifacts/models/best_model.pth",
    )
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load CIFAR-10 test dataset
    dataset = foz.load_zoo_dataset("cifar10", split="test")
    classes = dataset.default_classes
    image_paths, sample_ids = zip(*[(s.filepath, s.id) for s in dataset.iter_samples()])
    data_loader = make_cifar10_data_loader(image_paths, sample_ids, 16)

    # Make predictions
    with fo.ProgressBar() as pb:
        for imgs, sample_ids in pb(data_loader):
            predictions, probabilities, logits_ = predict(model, imgs.to(device))
            for sample_id, prediction, logits in zip(sample_ids, predictions, logits_):
                sample = dataset[sample_id]
                if sample.ground_truth.label != classes[prediction]:
                    sample.tags.append("mistake")
                sample["prediction"] = fo.Classification(
                    label=classes[prediction],
                    logits=logits,
                    confidence=float(probabilities[prediction]),
                )
                sample.save()

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
