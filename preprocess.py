from PIL import Image
import torch
from torchvision import transforms
import glob
import os

def preprocess_image(image_path, target_size=(700, 700), min_dim_size=360): 
   
    img = Image.open(image_path).convert("RGB")  

    if img.width < min_dim_size or img.height < min_dim_size:
        return None  

    preprocess = transforms.Compose([
        transforms.Resize(700),  
        transforms.CenterCrop(672), # this vit can only take input of this size
        transforms.ToTensor(),  # Convert to tensor (values between 0 and 1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained models (like ViT)
    ])

    img_tensor = preprocess(img)
    
    return img_tensor

def get_image_paths(folder_path, file_extension="*.jpg"):
    
    return glob.glob(os.path.join(folder_path, file_extension))
