# Self-Supervised Learning with Joint Embedding Predictive Architecture (JEPA)

## Learning by Observation: From Cats & Dogs to Tigers & Foxes

## Overview

This project explores the potential of the **Joint Embedding Predictive Architecture (JEPA)** for learning high-level representations through **self-supervised learning (SSL)**. Inspired by human cognitive abilities to generalize through observation, JEPA learns latent representations of data.

### Key Objectives
1. Train a simplified JEPA model using **unlabeled images of dogs and cats**.
2. Test the learned representations on **images of tigers and foxes**.
3. Analyze the generalization ability through **clustering and visualization** of the learned representations.

## Features
- Preprocessing pipeline for handling and augmenting image data.
- Training and testing of a JEPA-based SSL model.
- Visualization of learned latent space representations.
- Clustering to uncover inherent patterns in representations.

## Project Structure
JEPA/
├── main.py 
├── preprocess.py # Handles resizing, normaliztion
├── patcher.py # Extracts patches, randomly assigns target and context patches, adds positional embeddings 
├── encoder.py # loads pre-trained ViT
├── predictor.py # using MLP with one RELU layer to train context using target patches 



## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ManushKalwari/JEPA-Project.git
  
