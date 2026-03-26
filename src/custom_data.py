
#import libraries
import torch
from typing import List
import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets

#to show random image from the dataset
def show_random_image(image_path_list):
    # random.seed(42)                #for reproducibility

    # #get a list of all the image paths in the dataset
    #get a random image path
    random_image = random.choice(image_path_list)    
    
    #open the image
    img = Image.open(random_image)
    
    #get the class name from the image path
    class_name = random_image.parent.stem
    
    #show the image and class name
    print(f"Random image path: {random_image}")
    print(f"Image class: {class_name}")
    print(f"Image height: {img.height}") 
    print(f"Image width: {img.width}")   
    return img                                  #it is for viualize the image


# This method is used to display orginal image with its transformed image.
def plot_transformed_images(image_paths, transform, n=3, seed=42):

    random.seed(seed)                                             #it used for reproducablity, when we run the code giving exactly the same output.
    
    random_image_paths = random.sample(image_paths, k=n)           # Sample returns a list of arrays, have n length.
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2 , figsize=(12, 5))
            # Orginal image plot
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)



# The Method is used to display/show random images
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          row = None,
                          columns = None,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10 and n == 10:
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(12, 6))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [ height, width, color_channels]
        targ_image_adjust = targ_image.permute(1, 2, 0)
        
        # Plot adjusted samples
        plt.subplot(row, columns, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        plt.tight_layout()
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)