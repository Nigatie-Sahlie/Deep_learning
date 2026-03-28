# Creating Dataloaders
import os
import zipfile
import requests

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# The method for checking the data's folder and download if not exist.
def check_data(data_path, image_path):
    # Check if the image path exists or not.
    if image_path.is_dir():
        print(f"{data_path} directory already exists, skipping download.")
    else:
        print(f"{data_path} directory does not exist, creating one and downloading data.")
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Download the data
        with requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        ) as r:
            with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
                f.write(r.content)      
        # Unzip the data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            zip_ref.extractall(image_path),
        
        # Remove the zip file
        os.remove(data_path / "pizza_steak_sushi.zip")  


# Define the function to create dataloaders
def create_dataloaders(train_dir: str,
                      test_dir: str,
                      train_transform: transforms.Compose,
                      test_transform: transforms.Compose,
                      batch_size: int,
                      num_workers: int):
                      
    # Using imagefolder to create datasets
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transform)
    # get class names
    class_names = train_data.classes
    
    # Then turn images into dataloaders  or into batches.
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return train_dataloader, test_dataloader, class_names   