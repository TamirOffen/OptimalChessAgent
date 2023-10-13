"""
Configurations:

Note: you need to fill in the directories specific to your computer
"""

import torch

class Config:
    # Data loading and preprocessing
    raw_data_dir = "/home/tamiroffen/AI_Project/chess_games.csv"
    train_data_dir = "/home/tamiroffen/AI_Project/train_dataset.csv"
    valid_data_dir = "/home/tamiroffen/AI_Project/valid_dataset.csv"
    test_data_dir = "/home/tamiroffen/AI_Project/test_dataset.csv"

    chess_dataset_dir = "/home/tamiroffen/AI_Project/chess_games"
    eval_dataset_dir = "/home/tamiroffen/AI_Project/chess_evals"


    # Training hyperparameters
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-5

    # Paths for saving 
    runs_dir = "/home/tamiroffen/AI_Project/project/runs"
    saved_models_dir = "/home/tamiroffen/AI_Project/project/saved_models"

    # Other settings
    device = "cuda" if torch.cuda.is_available() else "cpu"



