
# from csv import writer

import torch
from Modularity.going_modular.engine import train_step, test_step
import tqdm

# This is updated script for training and evaluation model and it includes the SummeryWitter for logging the training and evaluation metrics to TensorBoard.
def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                epochs: int,
                device: torch.device,
                writer: torch.utils.tensorboard.SummaryWriter): # type: ignore
    
    result = {"train_loss": [],
              "train_accu": [],
              "test_loss": [],
              "test_accu": []}
    
    # Create a SummaryWriter to log the training and evaluation metrics to TensorBoard
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss, train_accu = train_step(model=model,
                                           dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           device=device)
        
        test_loss, test_accu = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        # Print waht is happening
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_accu: {train_accu:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_accu: {test_accu:.4f}")
        
        # Update the dictionary with the results
        result["train_loss"].append(train_loss)
        result["train_accu"].append(train_accu)
        result["test_loss"].append(test_loss)
        result["test_accu"].append(test_accu)

        # add accuracy and loss to tensorboard
        writer.add_scalar("train_accu", train_accu, epoch)
        writer.add_scalar("test_accu", test_accu, epoch)

        # Track the pyTorch model architecture
        writer.add_graph(model = model, 
                         input_to_model = torch.randn(32, 3, 224, 224).to(device)) 
        #   Close writer
        writer.close()

      # Return the result dictionary.
    return result