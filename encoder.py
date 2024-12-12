import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights


def load_vit_encoder(pretrained=True):
  
    vit_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)  # ViT base model (16x16 patches)
    vit_model.head = torch.nn.Identity()
    return vit_model



def extract_patch_embeddings(context_patches, target_patches, context_encoder, target_encoder):
 
    context_embeddings = []
    target_embeddings = []

    for patch in context_patches:
        #print(f"\nCONTEXT\nPatch shape before processing: {patch.shape}")
        patch = patch.unsqueeze(0)  # Add batch dimension
        #print(f"Patch shape after unsqueezing: {patch.shape}")
        context_embed = context_encoder(patch)
        #print(f"Context embed shape: {context_embed.shape}")
        context_embeddings.append(context_embed)

    for patch in target_patches:
        #print(f"\nTARGET\nPatch shape before processing: {patch.shape}")
        patch = patch.unsqueeze(0)  # Add batch dimension
        #print(f"Patch shape after unsqueezing: {patch.shape}")
        target_embed = target_encoder(patch)
        #print(f"Context embed shape: {target_embed.shape}")
        target_embeddings.append(target_embed)


    #print(f"\nNumber of context embeddings: {len(context_embeddings)}")

    #print("Context embeddings content:")
    #for i, embed in enumerate(context_embeddings):
        #print(f"Embedding {i}: {embed.shape}")

    #print(f"\nNumber of target embeddings: {len(target_embeddings)}")
    #print("Target embeddings content:")
    #for i, embed in enumerate(target_embeddings):
        #print(f"Embedding {i}: {embed.shape}")

    
    return context_embeddings, target_embeddings
