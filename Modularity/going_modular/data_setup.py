# Creating Dataloaders
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the function to create dataloaders
def create_dataloaders(train_dir: str,
                      test_dir: str,
                      transform: transforms.Compose,
                      batch_size: int,
                      num_workers: int ):
                      
    # Using imagefolder to create datasets
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)
    # get class names
    class_names = train_data.classes
    print(f"Class names: {class_names}")

    # Then turn images into dataloaders

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return train_dataloader, test_dataloader, class_names   