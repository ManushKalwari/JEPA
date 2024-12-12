import torch
from torchvision import transforms
import numpy as np
import random

def extract_patches(image_tensor, patch_size=(224, 224), grid_size=(3, 3)):
    """
    Extracts patches from the input image tensor.
    
    Parameters:
        image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).
        patch_size (tuple): Size of each patch (height, width), default (224, 224) since ViT needs that
        grid_size (tuple): The number of patches in each row and column (default is 3x3, 9 patches).
        
    Returns
        patches (torch.Tensor): Tensor of shape (num_patches, C, patch_height, patch_width).
    """
    C, H, W = image_tensor.shape
    assert H == W == 672, "The image must be 22xx224 pixels."

    # Calculate the number of patches in each dimension (H/W // patch_size)
    patches = []

    for i in range(grid_size[0]):  
        for j in range(grid_size[1]):  
            # Define start and end coordinates for each patch
            start_h = i * patch_size[0]
            end_h = (i + 1) * patch_size[0]
            start_w = j * patch_size[1]
            end_w = (j + 1) * patch_size[1]
            
          
            patch = image_tensor[:, start_h:end_h, start_w:end_w]
            patches.append(patch)
    
   
    patches_tensor = torch.stack(patches)
    
    return patches_tensor


def separate_patches(patches, num_target=4):
    """
    Randomly separates patches into context and target groups.

    Parameters:
        patches (torch.Tensor): A tensor of shape (9, C, H, W) containing image patches.
        num_target (int): Number of patches to use as target patches.

    Returns:
        context_patches (torch.Tensor): Tensor of context patches (5, C, H, W).
        target_patches (torch.Tensor): Tensor of target patches (4, C, H, W).
        target_positions (list): Indices of the target patches.
    """
    total_patches = patches.shape[0]
    #print(f"Total Patches: {str(total_patches)}")
    assert total_patches == 9, "There should be exactly 16 patches from the 400x400 image."

    # Randomly select indices for the target patches
    target_indices = random.sample(range(total_patches), num_target)
    context_indices = [i for i in range(total_patches) if i not in target_indices]

    # Separate the patches
    target_patches = patches[target_indices]
    context_patches = patches[context_indices]

    return context_patches, target_patches, target_indices, context_indices


def add_positional_encoding(embeddings, positions, d_model=1000):
    """
    Add positional encoding to the embeddings.
    
    Parameters:
    - embeddings: tensor of shape (batch_size, num_embeddings, embedding_dim)
    - positions: list or tensor of shape (num_embeddings,) containing the target positions
    - d_model: the dimension of the embedding (default 1000 in your case)
    
    Returns:
    - Embeddings with added positional encoding
    """

    # If embeddings is a list, stack them into a single tensor
    if isinstance(embeddings, list):
        embeddings = torch.stack(embeddings)  # Combine list of tensors into a single tensor

    embeddings = embeddings.squeeze(1)
    #print(f"Shape of Embeddings: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Convert positions from list to tensor
    positions = positions.clone().detach() if isinstance(positions, torch.Tensor) else torch.tensor(positions, dtype=torch.long)
    #print("Positions: ",positions)
    
    num_embeddings = embeddings.shape[0]

    # Generate the positional encodings using sine and cosine functions
    position_encodings = torch.zeros((num_embeddings, d_model))


    
    for pos in range(num_embeddings):
        for i in range(d_model):
            divisor = 10000 ** (i / d_model)
            if i % 2 == 0:
                position_encodings[pos, i] = torch.sin(positions[pos] / divisor)
            else:
                position_encodings[pos, i] = torch.cos(positions[pos] / divisor)
    

    position_encodings = position_encodings.squeeze(0)
    #print(f"Shape of sine-encoded Positions: {position_encodings.shape}, dtype: {position_encodings.dtype}")
    
    embeddings_with_pos = embeddings + position_encodings

    #print(f"Final Shape: {embeddings_with_pos.shape}, dtype: {embeddings_with_pos.dtype}")
    
    return embeddings_with_pos




