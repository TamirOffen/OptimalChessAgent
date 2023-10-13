"""
Preprocessing Module:

Place your preprocessing code in a separate module or Python file (e.g., preprocess.py).
Define functions or classes for data preprocessing, such as image resizing, normalization, and augmentation.
This module should be responsible for preparing your raw data for training.
"""


# imports:
import numpy as np  
import re
import pandas as pd
import gc
from sklearn.preprocessing import StandardScaler

# credit: Moran Reznik 
# Column Indexes Mapping:
"""
notice that columns are letters and rows are numbers in a chess board
"""
letter_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
num_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}


# credit: Moran Reznik 
# Chess board to tensor:
"""
idea 1: assign a unique number to each type of piece = bad idea

why? does not reflect the relationship of the pieces in a good way to learn. example: white pawn = 1, black pawn = 2; so that means black pawn is worth more than a white pawn...

idea 2: layers. CNN takes in a 3D input, each layer is a type of piece. black will be -1, whites will be 1.
"""
def create_rep_layer(board, piece_type):
  # board is 8x8
  s = str(board) # string representation of board, . = whitespace, upper = white, lower = black, 8x8
  s = re.sub(f'[^{piece_type}{piece_type.upper()} \n]', '.', s) # replace anything but our type of piece with a .
  s = re.sub(f'{piece_type}', '-1', s) # change black type pieces to -1
  s = re.sub(f'{piece_type.upper()}', '1', s) # change white type pieces to 1
  s = re.sub(f'\.', '0', s) # change whitespaces to 0s
  # s a string, visually is 8x8 with {0,-1,1} s.t. 1,-1 represent piece type.

  # want to convert s into an int matrix
  board_mat = []
  for row in s.split('\n'):
    row = row.split(' ')
    row = [int(x) for x in row]
    board_mat.append(row)

  return np.array(board_mat)


# credit: Moran Reznik 
# board is a chess library board
def board_to_rep(board):
  # board is 8x8 representation of a board state in chess
  pieces = ['p','r','n','b','q','k']
  layers = [] # will be 6 layers (one for each type of piece)
  for piece in pieces:
    layers.append(create_rep_layer(board, piece))
  board_rep = np.stack(layers) # 3D tensor
  return board_rep


# credit: Moran Reznik 
# chess move to tensor:
"""
idea 1: traditional multi-class clas. Bad idea because there are many moves possible in chess (not all legal for every state ofc which means each board has a different amount of legal moves). Also, the index of the move does not have any info for learning, which does not use CNN's ability to caputre patterns etc.

idea 2: represent a move with a 'to matrix' and a 'from matrix'. to matrix is where to move the piece, and from matrix tells us which piece to move.
"""
# move needs to be in san format
# returns from, to
def move_to_rep(move, board):
  board.push_san(move).uci() #convert the board from san to uci
  # uci format example: d4e5 = take piece from col-d, row-4 to col-e, row-5
  move = str(board.pop())

  from_output_layer = np.zeros((8,8))
  from_row = 8 - int(move[1])
  from_column = letter_to_num[move[0]]
  from_output_layer[from_row, from_column] = 1

  to_output_layer = np.zeros((8,8))
  to_row = 8 - int(move[3])
  to_column = letter_to_num[move[2]]
  to_output_layer[to_row, to_column] = 1

  return np.stack([from_output_layer, to_output_layer])


# credit: Moran Reznik 
# Game to Moves
"""
raw dataset looks like this: (san notation)
1. d4 d5 2. Nf3 Nf6 3. Bf4 c6 etc...
"""
# note: excludes last move
def create_move_list(s):
  return re.sub('\d*\. ', '', s).split(' ')[:-2]


# credit: Moran Reznik 
# returns a chess data df, with long enough games (>20 moves) and high elo games (>2000)
def preprocess_chess_data(data_path):
    # Load the raw data
    chess_data_raw = pd.read_csv(data_path, usecols=['AN', 'WhiteElo'])

    # Filter and preprocess the data
    chess_data = chess_data_raw[chess_data_raw['WhiteElo'] > 2000]
    del chess_data_raw
    gc.collect()
    chess_data = chess_data[['AN']]
    chess_data = chess_data[~chess_data['AN'].str.contains('{')]
    chess_data = chess_data[chess_data['AN'].str.len() > 20]

    return chess_data


# returns a eval data df, with only numbers in evals, i.e. checkmate in n moves is made into a very large +- number
def preprocess_eval_data(data_path):
    df = pd.read_csv(data_path)

    # replace non numerical values in the Evaluation column (#+[num] or #-[num] symbolizing checkmate in num moves)
    column_name = "Evaluation"
    pattern = r'#[\+]?\d+' # replace with a very large positive number
    mask = df[column_name].str.contains(pattern, regex=True)
    df.loc[mask, column_name] = df.loc[mask, column_name].str.replace(pattern, '2500', regex=True)
    pattern = r'#[\-]?\d+' # replace with a very large negative number
    mask = df[column_name].str.contains(pattern, regex=True)
    df.loc[mask, column_name] = df.loc[mask, column_name].str.replace(pattern, '-2500', regex=True)
    df[column_name] = df[column_name].astype("int64")

    # normalize data (gaussian)
    scaler = StandardScaler()
    scaler.fit(df[['Evaluation']])
    df['Evaluation_std'] = scaler.transform(df[['Evaluation']])

    return df










