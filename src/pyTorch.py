import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report


#To calculate the total time taken for the spacific code.
def time_calculator(start, end):
    return print(f"The total time taken is: {(end - start)*1000:.2f} mini seconds.")

#model class
class classification_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5) #have five hidden neurons or unit
        self.layer2 = nn.Linear(in_features=5, out_features=1)
    def forward(self, x):
        return self.layer2(self.layer1(x))

class classification_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))
    
#This calculate the accuracy of the model
def man_accuracy(y_true, y_pred):
    y_round = torch.round(torch.sigmoid(y_pred))
    correct = torch.eq(y_round, y_true).sum().item()
    return f"{(correct / len(y_true)) * 100:.2f}%"

#scatter plot
def scatter_plot(X_train, y_train, x_test = None, y_test = None, y_pred = None, title="Scatter Plot"):
    plt.figure(figsize=(15, 7))
    plt.scatter(X_train, X_train, c=y_train, cmap=plt.cm.RdYlBu, label="Train")
    if x_test is not None and y_test is not None:
        plt.scatter(x_test, x_test, color = "orange", label="Test")
    if y_pred is not None:
        plt.scatter(x_test, y_pred, color = "green", label="Predicted")
    plt.title(title)
    plt.legend()
    plt.show()

# #for plot
def plot(X, Y, title="Plot"):
    plt.figure(figsize=(20, 10))
    plt.plot(X, Y, color="blue", marker="o", linestyle="--")
    plt.title(title)
    plt.show()

#the training and test loop function
"""The train loop and test loop function are reusable functions for training and testing a PyTorch model. They take in a model, data, loss function, accuracy function and device and perform the training and testing loop respectively. They also print the average loss and accuracy for each epoch."""
def train_loop_fnc(model: torch.nn.Module,
        data: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accuracy_fun,
        device: torch.device):
    
    train_loss, train_acc = 0, 0
    model.to(device)    #the model to proper device
    
    for batch, (x, y) in enumerate(data):
        #the data should be to the proper devece
        x, y = x.to(device), y.to(device)
        
        model.train()

        #1. forward pass
        y_pred = model(x)
        train_acc += accuracy_fun(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        #2. calculate the loss function 
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        

        #3. zero gra
        optimizer.zero_grad()

        #4. backward pass
        loss.backward()

        #5. update the optimize
        optimizer.step()
        
    ave_train_loss = train_loss / len(data)
    av_train_accu = train_acc / len(data)
    print(f"Train loss: {ave_train_loss:.5f} | Train Accuracy: {av_train_accu:.2f}%\n")

#for test loop
def test_loop_fnc(
        model: torch.nn.Module,
        data: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        accuracy_fun,
        device: torch.device):
    
    test_loss, test_acc = 0, 0
    model.to(device)    #the model to proper device
    
    with torch.inference_mode():
        for batch, (x, y) in enumerate(data):
            #the data should be to the proper devece
            x, y = x.to(device), y.to(device)
            
            model.eval()

            #1. forward pass
            test_pred = model(x)
            test_acc += accuracy_fun(y_true=y, y_pred=test_pred.argmax(dim=1))
            
            #2. calculate the loss function 
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
        
    ave_test_loss = test_loss / len(data)
    av_test_accu = test_acc / len(data)
    print(f"Test loss: {ave_test_loss:.5f} | Test Accuracy: {av_test_accu:.2f}%")


#****************Computer vision part***************************8
torch.manual_seed(42)
def model_eval(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """This function is for evaluating the pyTorch model on a given data_loader with a given loss function and accuracy function.
    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

#Calculating confusion matrix
def evaluate_cnn_metrics(model, test_loader, device):
    """
    Evaluate a PyTorch CNN model using:
    - Confusion Matrix
    - Precision
    - Recall
    - F1 Score

    Args:
        model: trained PyTorch model
        test_loader: DataLoader for test dataset
        device: 'cuda' or 'cpu'

    Returns:
        dict containing all evaluation metrics
    """

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Precision, Recall, F1 (macro = treats all classes equally)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # Detailed per-class report
    report = classification_report(y_true, y_pred)

    results = {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report
    }

    return results


#visualization of confusion matrix
def plot_confusion_matrix(cm, class_names=None):    
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,)
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.show()




