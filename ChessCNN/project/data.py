"""
Data Classes and Data Loaders:

Create a separate module (e.g., data.py) to define your data classes and data loaders.
Define classes or functions that encapsulate your dataset and data loading logic.
This module should provide convenient access to training and validation data.
"""


# imports:
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocess import *
import chess
from config import Config
import pandas as pd
import numpy as np


# PyTorch Dataset / Dataloader
"""
1. __init__ = raw dataset
2. __len__ = number of training examples in the dataset
3. __getitem__ = returns training examples at index
"""

# credit: Moran Reznik https://www.youtube.com/watch?v=aOwvRvTPQrs 
class ChessDataset(Dataset):
  def __init__(self, games, length=40_000):
    super(ChessDataset, self).__init__()
    self.games = games
    self.length = length

  def __len__(self):
    return self.length #samples a random 40,000 games from the 800,000+
    # num of batches in one epoch = ceil(40000 / batch_size)

  # ( x=state of board 6x8x8, y=(from, to) )
  def __getitem__(self, index):
    game_idx = np.random.randint(self.games.shape[0]) #random game
    random_game = self.games.values.flatten()[game_idx]
    moves = create_move_list(random_game)
    game_state_idx = np.random.randint(len(moves)-1) #random move in that game
    next_move = moves[game_state_idx] 
    moves = moves[:game_state_idx] 
    board = chess.Board()
    for move in moves:
      board.push_san(move)
    x = board_to_rep(board)
    y = move_to_rep(next_move, board)
    # game_state is odd <=> black's turn
    if game_state_idx % 2 == 1:
      x *= -1
    return x, y, game_state_idx


class EvalDataset(Dataset):
    def __init__(self, games, length):
        super(EvalDataset, self).__init__()
        self.games = games
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        games = self.games
        game_idx = np.random.randint(len(games))
        board_fen = games.values[game_idx][0]
        board_eval = games.values[game_idx][2]
        board = chess.Board(board_fen)
        x = board_to_rep(board)
        y = board_eval
        return x, y
            



