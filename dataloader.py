import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_dataloaders(batch_size: int = 32, num_workers: int = 2, target_size: int = 32) -> tuple[DataLoader, DataLoader]:
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    resize = T.Resize((target_size, target_size)) if target_size else T.Lambda(lambda x: x)

    train_transform = T.Compose(
        [
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.RandomPerspective(distortion_scale=0.5, p=0.5),
            T.RandomHorizontalFlip(),
            resize,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    val_transform = T.Compose(
        [
            resize,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
