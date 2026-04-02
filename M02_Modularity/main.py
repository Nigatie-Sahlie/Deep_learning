
import os
import torch
from going_modular import data_setup, engine, model_builder, utils
from typing import cast

from torchvision import transforms



def main(train_dir: str, 
         test_dir: str, 
         save_dir: str = "model"):
    
    # Setup hyperparameters
    NUM_EPOCHS = 8
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
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
    

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
                                        train_dir=train_dir,
                                        test_dir=test_dir,
                                        train_transform=train_transform,
                                        test_transform=test_transform,
                                        batch_size=BATCH_SIZE,
                                        num_workers=cast(int, os.cpu_count())
                                    )
    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # Start training with help from engine.py
    train_result = engine.train_loop(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epoches=NUM_EPOCHS,
                device=device)
    
    return model, test_dataloader, class_names, train_result
    
