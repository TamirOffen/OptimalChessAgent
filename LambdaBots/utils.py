
# partial credit: Moran Reznik https://www.youtube.com/watch?v=aOwvRvTPQrs


# imports:
import chess
import numpy as np
import torch
from preprocess import *
# from config import Config

# can use this func to find k-move forced mate (recursion)
def check_mate_single(board):
  board = board.copy()
  legal_moves = list(board.legal_moves)
  for move in legal_moves:
    board.push_uci(str(move))
    if board.is_checkmate():
      move = board.pop()
      return move
    _ = board.pop()


"""
input: list of nums
output: probablity distribution
"""
def distribution_over_moves(vals):
  probs = np.array(vals)
  probs = np.exp(probs)
  probs = probs / probs.sum() #makes a legal prob. dist.
  probs = probs ** 3 #further separates high and low probs
  probs = probs / probs.sum()
  return probs


"""
choose_move function takes in a chess board state, predicts a move using a trained model (saved_model),
and selects a move to play based on the predicted values and probabilities. The function incorporates
considerations such as checkmate moves and probabilistic selection based on the saved_model's predictions.
"""
# returns None if board is in a terminal state
def choose_move(board, color, model):
  # list of legal moves available to player (color) based on board
  legal_moves = list(board.legal_moves) 
  if not legal_moves:
      # board is in a terminal state.
      return None

  # config = Config()
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # check if there is an immediate checkmate move available
  # TODO: add recursive k-step forced mate ?
  move = check_mate_single(board)
  if move is not None:
    return move

  # transform the board so that it can be used by the CNN 
  board_rep = board_to_rep(board)
  x = torch.Tensor(board_rep).float().to(device)
  if color == chess.BLACK:
    x *= -1 # flips the board if player is black
  x = x.unsqueeze(0) # adds a dim (batch_size=1), shape=(1,2,8,8)
  move = model(x) # model prediction
  move = move.squeeze(0) # removes batch dim, shape=(2,8,8)
  from_matrix, to_matrix = move[0], move[1]


  vals = [] # values inside of from matrix corresponding to the legal moves
  # starting square of each legal move, i.e. legal_move=e2e4, starting square=e2
  froms = [str(legal_move)[:2] for legal_move in legal_moves]
  froms = list(set(froms)) # remove duplicate elements
  for from_ in froms:
    board_pos = (8-int(from_[1]), letter_to_num[from_[0]])
    val = from_matrix[board_pos]
    vals.append(val)
  vals = torch.tensor(vals).to(device)
  probs = distribution_over_moves(vals.cpu())
  if np.any(np.isnan(probs)):
      probs = np.nan_to_num(probs, nan=0.0)
      probs /= probs.sum()  # normalize the probs
  # probs still contains nans, we will return a random move
  if np.any(np.isnan(probs)):
      rand_move_idx = np.random.randint(len(legal_moves))
      return legal_moves[rand_move_idx]

  # if not froms:
  #     # Game is in a terminal state.
  #     return None

  # randomly select a square in froms based on probabilities 'prob'.
  # this will be the piece that we will move.
  chosen_piece = str(np.random.choice(froms, size=1, p=probs)[0])

  # values in to_matrix corresponding to chosen_piece
  vals = []
  for legal_move in  legal_moves:
    from_ = str(legal_move)[:2]
    if from_ == chosen_piece:
      to = str(legal_move)[2:]
      board_pos_to = (8-int(to[1]), letter_to_num[to[0]])
      val = to_matrix[board_pos_to]
      vals.append(val)
    else:
      vals.append(0)

  vals = torch.tensor(vals).to(device)
  chosen_move = legal_moves[vals.argmax()]
  return chosen_move



def tensor_to_board(board_tensor):
    board = chess.Board()
    board.clear_board()
    piece_chars = ['p', 'r', 'n', 'b', 'q', 'k']

    for i, piece_char in enumerate(piece_chars):
        black_indices = (board_tensor[i] == -1)
        white_indices = (board_tensor[i] == 1)
        for row in range(8):
            for col in range(8):
                if black_indices[row][col]:
                    board.set_piece_at(chess.square(col, 7 - row), chess.Piece.from_symbol(piece_char))
                elif white_indices[row][col]:
                    board.set_piece_at(chess.square(col, 7 - row), chess.Piece.from_symbol(piece_char.upper()))

    return board


