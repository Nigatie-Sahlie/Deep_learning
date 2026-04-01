import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

def evaluation(model, test_data, device):
    # Turn the model in evaluation mode
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for image, label in test_data:
            # Turn the dataset to the approprate device
            images = image.to(device)
            labels = label.to(device)
            
            # Make prediction
            logit = model(images)
            _, predictioin = torch.max(logit, 1)

            # To CPU memory 
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictioin.cpu().numpy())
        
        # Conver the data to numpy array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
    # From Confution matricx
    confu_matri = confusion_matrix(y_true=y_true, 
                                   y_pred=y_pred)
    precision = precision_score(y_pred=y_pred, y_true=y_true, average="macro")
    recall = recall_score(y_true = y_true, y_pred = y_pred, average="macro")
    f1  = f1_score(y_true,  y_pred, average="macro")
    classi_report = classification_report(y_true, y_pred)
    print(classi_report)

    eval_metrics = {
        "precision": f"{precision*100:.2f}%",
        "Recall": f"{recall*100:.2f}%",
        "f1_score": f"{f1*100:.2f}%"
    }
    return eval_metrics, confu_matri

# Plot the evaluation of the model
def plot_conf_metrics(confution_matr, class_name:str):
    plt.figure(figsize=(15, 8))
    sns.heatmap(confution_matr, annot=True, 
                xticklabels=class_name, 
                yticklabels=class_name)
    plt.xlabel("predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.show()