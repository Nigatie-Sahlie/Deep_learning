
#import libraries
import random
from PIL import Image


#to show random image from the dataset
def show_random_image(image_path_list):
    random.seed(42) #for reproducibility

    # #get a list of all the image paths in the dataset
    # jpg_image_list = [img for img in image_path_list if img.suffix == ".jpg"]
    #get a random image path
    random_image = random.choice(image_path_list)   #it choice one random element from the list
    
    #open the image
    img = Image.open(random_image)
    
    #get the class name from the image path
    class_name = random_image.parent.stem
    
    #show the image and class name
    print(f"Random image path: {random_image}")
    print(f"Image class: {class_name}")
    print(f"Image height: {img.height}") 
    print(f"Image width: {img.width}")   
    return img    #it is for viualize the image

