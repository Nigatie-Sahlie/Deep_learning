import torch
from torchvision import transforms


# Do preprocessing and setup for transfer learning notebook
def preprocess():
    train_transform = transforms.Compose([
        transforms.Resize((264, 264)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((264, 264)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
    ])
    return train_transform, test_transform