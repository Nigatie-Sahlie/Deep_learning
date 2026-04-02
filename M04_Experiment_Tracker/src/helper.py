import torch
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter

# Random seed for reproduciblity
def set_seed(seed: int = 42):

    torch.manual_seed(seed)      # This is for CPU reproducibility 
    torch.cuda.manual_seed(seed)  # This is for current GPU reproducibility
    torch.cuda.manual_seed_all(seed)  # This is for all GPU reproducibility (if you are using multi-GPU)




# Let's create helper function to control the log_dir of the SummeryWriter() instance.
def create_writer(root_dir: str,
                  experiment_name: str,
                  model_name: str,
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter(): # type: ignore
    
    """The function are doing to save the log and track on the desired(targeted) custom directory.
    Args:
        experiment_name - it is string value and the name of the experiment.
        model_name = it is also string data type and indicates the name of the model used.
        extra - Other information incase needed to more discribe the experiment.
        
    return:
        A SummaryWriter instance with the log_dir set to the desired directory.
    """
    
    # Get the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")     # It records the current date and time when the code is run.
    
    if extra:
        log_dir = os.path.join(root_dir, "logs", experiment_name, model_name, extra, timestamp)  # This is the directory where the logs will be saved.
    else:
        log_dir = os.path.join(root_dir, "logs", experiment_name, model_name, timestamp)          # This is the directory where the logs will be saved.
    print(f"[INFO] Creating a new SummaryWriter with log_dir: {log_dir}")

    # Return the summery writer instance.
    return SummaryWriter(log_dir=log_dir)