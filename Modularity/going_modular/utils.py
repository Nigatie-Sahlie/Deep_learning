# now creating the function to save the model
import torch 
from pathlib import Path

# create function
def save_model(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str):
    
    # create directory from the target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should end with .pth or .pt"
    model_save_path = target_dir_path / model_name

    # then save the model state dict
    print(f"Saving the model to: {model_save_path}")
    torch.save(obj=model.state_dict(),  f=model_save_path)
    print("Model saved successfully!")