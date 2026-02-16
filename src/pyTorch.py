import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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

