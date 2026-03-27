# Import libraries
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# creating training loop
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    # model in train mode
    model.train()

    # train loss and train accuracy
    train_loss, train_acc = 0, 0

    # loop through data loader
    for batch, (X, y) in enumerate(dataloader):
        x, y = X.to(device), y.to(device)

        #forward pass
        y_pred = model(x)

        # calculate and accomulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #Optimizer
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        #optimizer step
        optimizer.step()

        # calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    #Adjest metrics and return
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# test loop
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
 
    # put model in evaluation mode
    model.eval()

    # test loss and test accuracy
    test_loss, test_acc = 0, 0

    # turn on inference context manager
    with torch.inference_mode():
        # Loop through dataloader
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            test_pred_logit = model(x)

    
            # calculate and accumulate loss
            loss = loss_fn(test_pred_logit, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy metrics across all batches
            test_pred_label = test_pred_logit.argmax(dim=1)
            test_acc += ((test_pred_label == y).sum().item()/len(test_pred_label))
    
    # Adjest metrics to get average loss
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# final training loop
def train_loop(model:torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                epoches: int,
                device: torch.device) -> Dict[str, List]:
    
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_accu": []
               }
    for epoch in tqdm(range(epoches)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_accu"].append(test_acc)
    return results