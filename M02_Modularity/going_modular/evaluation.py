import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple

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


# Predict on a target image with a target model
# Function created in: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set
def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None):    # type: ignore


    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target 
    # model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)    # type: ignore

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image)

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Actual: {Path(image_path).name} | Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)