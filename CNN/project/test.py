"""
Testing Scripts:

"""

from config import Config
from preprocess import *
from utils import choose_move, tensor_to_board
import chess
import torch
import numpy as np


"""
from accuracy:
accuracy of the model to choose the right piece to move.

to accuracy:
we want to test piece move accuracy.
therefore we will have 6 types of accuracies, one for each piece type: [p, r, n, b, q, k].
we will also include an overall piece move accuracy.

implementation:
random game, random move. check if next move is what the model would have chosen. 
update accuracy according to piece type.

"""

# board is of shape (6,8,8)
# from_to_matrix is of shape (2,8,8)
# model_move is of shape (2,8,8) (from, to matrix)
# returns a tuple: 
#   (True if chose right piece to move False o.w., piece_type, True if correct move False o.w.)
def move_acc(board, from_to_matrix, model_move):
    from_mat, to_mat = from_to_matrix[0], from_to_matrix[1]
    model_from, model_to = model_move[0], model_move[1]
    
    # from acc.    
    indices = torch.nonzero(from_mat)
    i, j = indices[0]  # location of piece that dataset will move
    i, j = i.item(), j.item()
    indices = torch.nonzero(model_from)
    x, y = indices[0]  # location of piece that model will move
    x, y = x.item(), y.item()
    if x != i or y != j:
        # model did not choose the right piece to move
        return (False, -1, False)

    # to acc.
    pieces = ['p','r','n','b','q','k']
    piece_type = -1
    for t in range(board.size(0)):
        curr_slice = board[t, :, :]
        if curr_slice[i][j] == 1 or curr_slice[i][j] == -1:
            piece_type = pieces[t]
            break
    if piece_type == -1:
        print("ERROR WHILE TESTING: in get_piece_type_by_move(...), could not find piece type")
        return (False, -1, False)
    indices = torch.nonzero(to_mat)
    i, j = indices[0]  # to location of piece of dataset move
    i, j = i.item(), j.item()
    indices = torch.nonzero(model_to)
    x, y = indices[0]  # to location of piece of model move
    x, y = x.item(), y.item()
    if x != i or y != j:
        # model did not choose the right dest.
        return (True, piece_type, False)
    else:
        # model did choose the right dest.
        return (True, piece_type, True)

    

# returns a 3-tuple:
#     from accuracy, [p, r, n, b, q, k] accuracies, and overall piece accuracy
def test_piece_accuracy(model, data_loader, num_epochs=1):
    print(f"Testing piece accuracy of model, over {num_epochs} num of epochs")
    piece_correct_counts = [0] * 6
    piece_seen_counts = [0] * 6
    piece_types = ['p','r','n','b','q','k']
    from_correct, from_total = 0, 0
    config = Config()
    for epoch in range(num_epochs):
        for batch_num, batch in enumerate(data_loader):
            x, y, game_states = batch
            batch_size = x.size(0)
            for i in range(batch_size):
                board_pos = x[i]
                ideal_move = y[i]
                curr_player = chess.BLACK if game_states[i] % 2 == 1 else chess.WHITE
                if curr_player is chess.BLACK: # TODO: Added check if fine??
                    continue
                raw_model_move = choose_move(tensor_to_board(board_pos), curr_player, model)
                if raw_model_move is None:
                    continue 
                model_move = move_to_rep(str(raw_model_move), tensor_to_board(board_pos))
                model_move = torch.from_numpy(model_move)
                (model_chose_right_piece, piece_type, model_chose_correct_move) = move_acc(board_pos, ideal_move, model_move)
                from_total += 1
                if model_chose_right_piece is False: 
                    continue
                from_correct += 1
                index = 0
                if piece_type in piece_types:
                    index = piece_types.index(piece_type)
                else:
                    print(f"ERROR WHILE TESTING: in test_piece_accuracy(...), '{piece_type}' not found in the list.")
                    continue
                piece_seen_counts[index] += 1
                if model_chose_correct_move:
                    piece_correct_counts[index] += 1

    from_accuracy = from_correct / from_total
    # False if piece not seen at all
    piece_accuracies = [correct / seen if seen != 0 else False for correct, seen in zip(piece_correct_counts, piece_seen_counts)]
    weighted_sum = sum(accuracy * seen for accuracy, seen in zip(piece_accuracies, piece_seen_counts))
    total_weight = sum(piece_seen_counts)
    overall_piece_accuracy = weighted_sum / total_weight if total_weight != 0 else 0.0

    return (from_accuracy, piece_accuracies, overall_piece_accuracy)
                
    
            
            



