# AI Project - CS236502
Spring 2023

| Student | Student ID | Email | 
| ---     | --- | --- |
| Aron Klevansky| 941190845 | klevansky@campus.technion.ac.il |
| Niv Ostroff | 212732101 | nivostroff@campus.technion.ac.il |
| Tamir Offen | 211621479 | tamiroffen@campus.technion.ac.il |

## Research Question: 

    What is the optimal tradeoff between genetically tuned handcrafted features vs. neural trained features when creating a chess playing agent?


## Abstract:

In this paper, we were interested in investigating the effectiveness of different AI techniques in building chess engines. Primarily, we were interested in:

    1. Determining the effectiveness of deep learning techniques such as Convolutional Neural Networks

    2. Determining the effectiveness of handcrafted heuristics that are trained using genetic algorithms

    3. The effect of combining these two methods, and the weighted trade-off between them. 


### The CNN: 
From our research on Convolutional Neural networks, we found them to be very good at the beginning of chess games (as they memorise opening theory very well). However, they become less and less reliable the longer the game goes on. This is mainly due to the sheer volume of possible chess positions, and the inability of training data to cope with all possibilities. Additionally, we found that CNNs are very good at understanding good piece placement for short-range pieces such as knights and pawns, but they struggle for more long range pieces such as rooks and queens. This makes sense due to the small convolutional window that is generated through use of a CNN. Finally, we found that while the CNN was able to beat simple bots (such as SxRandom), it would often blunder away pieces meaning that on its own it was not good enough against stronger bots (such as CHoMPBot). 

### The Genetically Tuned Handcrafted Heuristics:
From our research on handcrafted algorithms and genetic heuristics, we found that while the evolutionary process used was helpful in establishing a basis of what the value of each of the parameters should be, the bias of the initial training sets of random positions meant that there were certain inconsistencies in the parameter values that arose. For example, the first organism that we trained believed that a rook was worth more than a queen because there were more positions where the rook was the piece that moved than there were positions where it was the queen that had to move. This inconsistency in the value of material was solved through the process of co-evolution. By selecting for organisms that valued material more, we managed to achieve a final organism that had more reasonable parameter values and blundered less. However, while the organism was able to beat the Lichess bot CHoMPBot, it struggled immensely against sargon-1ply. 

### The Weighted Tradeoff between the CNN and Genetically Tuned Heuristics:
Finally, when looking at the optimal tradeoff between the CNN and the genetically tuned heuristics, we found that the best $\lambda$ value was 0.5, dead in the centre between the genetic heuristic and the CNN! This $\lambda$ value was based on our experiments conducted on an external dataset, as well as the sets of games we had between bots using different $\lambda$ values. Indeed, we find the fact that the final model was able to beat bots such as sargon-1ply and CHoMPBot with both colours when both original models created were not able to achieve such a feat indicative of the potential that hybrid chess engine models have.
 

## Project Structure:

*Note: credit to the original author is written inside each file. If no comment, then written by Aron, Niv, and Tamir.*

### `ChessEngineOptimization.pdf`
- Paper explaining our project in detail.

### `libs.txt`
- List of Python packages required for the project. 

### `/CNN`
- Chapter 3.1 - ChessNet training and implementation
- `/CNN/ChessDataset.ipnyb`: Notebook for downloading the datasets used for the training and evaluation of ChessNet and ChessPosEvalNet. Note: `/CNN/kaggle.json` needs to be filled in to allow the notebook to accesss kaggle.
- `/CNN/project`
  - Contains the files for the CNN part of the project.
  - `/CNN/project/complex_cnn/complex_model.py`: pytorch model code for module, ChessPosEvalNet, and ChessNet.
  - `/CNN/project/runs`: tensorboard data collected while training. used for displaying training graphs of the models.
  - `/CNN/project/saved_models`: saved pytroch models for ChessPosEvalNet and ChessNet. Because of ChessPosEvalNet model size, the trained model can be found on [drive](https://drive.google.com/file/d/1LY1WybdkJewNCGMd1gjsbgnWiLPveL_v/view?usp=sharing).
  - `/CNN/project/ServerBashScripts`: bash scripts used to run the training files on the Technion Lambda server.
  - `/CNN/project/TrainingOutputs`: outputs of training.
  - `/CNN/project/board_eval_tests.ipynb`: tests for ChessEvalPosNet model after training.
  - `/CNN/project/ChessNet_final_tests.ipynb`: tests for ChessNet model after training.
  - `/CNN/project/config.py`: configurations, note: need to fill in specific to user.
  - `/CNN/project/data.py`: data classes and data loaders.
  - `/CNN/project/FinalModelExperiments.ipynb`: random experiments on the ChessNet model after 1 day of training. Includes a game of ChessNet vs. itself.
  - `/CNN/project/overleaf_printouts.ipynb`: code for some of the print outs in the paper.
  - `/CNN/project/preprocess.py`: functions used for preprocessing the chess data.
  - `/CNN/project/test.py`: testing scripts for ChessNet.
  - `/CNN/project/train_board_eval.py`: ChessEvalPosNet training script.
  - `/CNN/project/train_complex.py`: ChessNet training script.
  - `/CNN/project/train_tune_complex.py`: ChessNet hyperparameter tuning training script.
  - `/CNN/project/utils.py`: miscellaneous functions, notably: choose_move and tensor_to_board funcs.

### `/GeneticAlgorithm`
- Chapter 3.2 - Implementation of the Genetically Trained Heuristic
- `/GeneticAlgorithm/GA_training.ipynb`: Almost the same script as GA_training.py. This file has some headers that make the code segments easier to read. It is a recommended place to start when looking at the code for the genetic algorithm.
- `/GeneticAlgorithm/GA_training.py`: Script used to run regular genetic algorithm (no coevolution) on the LAMBDA server. Most of the code is very similar to that used in GA_training.ipynb. Training is done using the database lichess_elite_2022-12.
- `/GeneticAlgorithm/GA_runner.sh`: Bash file used for running GA_training.py on the LAMBDA server. 
- `/GeneticAlgorithm/coevolution_training.py`: Script used to run coevolution on the server. Most of the code is very similar to that used in GA_training.ipynb.
- `/GeneticAlgorithm/coevolution_training.sh`: Bash file used for running coevolution_training.py on the LAMBDA server. 

### `/LambdaBots`
- Chapter 4 - Experimental Methodology regarding the weighted trade-off between the CNN and the genetically tuned heuristics
- Chapter 5 - Experimental Results and Analysis of the weighted trade-off between the CNN and the genetically tuned heuristics
- `/LambdaBots/lambda_battle_arena.ipynb`: File used to run the matches between different lambda bots when conducting the tradeoff between the CNN and Genetic Algorithms
- `/LambdaBots/lambda_external_test.ipynb`: File used to run the tests of the differnet lambda bots on the external dataset lichess_elite_2022-01
- `/LambdaBots/saved_models`: contains the saved pytorch model ChessNet that is used as the CNN part of the weighted tradeoff.
- `/LambdaBots/preprocess.py`: auxilliary file used to load the CNN into the Jupyter notebooks
- `/LambdaBots/utils.py`: auxilliary file used to load the CNN into the Jupyter notebooks
- `/LambdaBots/complex_model.py`: auxilliary file used to load the CNN into the Jupyter notebooks

---

## User Guide:

To get started with the project, follow these steps (linux/mac):

### Environment Install: 
1. install [conda](https://www.anaconda.com/download).
2. create a conda env for the project:
   
```bash
conda create --name CNN_GenAlgo python=3.9
conda activate CNN_GenAlgo
```

3. install the package requirements:

```bash
pip install -r libs.txt
```

### Using the CNN:
- Change directory to `/CNN` and open the jupyter environment, usually with the command jupyter lab.
- Downloading the Datasets:
  - Get a `kaggle.json` file from [kaggle](https://www.kaggle.com/docs/api) and place it in `/CNN`
  - Run the cells in `/CNN/ChessDataset.ipynb` to download chess_games.csv and random_evals.csv.
- CNN Training:
  - To train a ChessNet model, run `/CNN/project/train_complex.py`.
  - To train a ChessPosEvalNet model, run `/CNN/project/train_board_eval.py`. 
  - To perform hyper paramter tuning of of the batch size and learning rate, run `/CNN/project/train_tune_complex.py`.
- How to Make Predictions with the Models:  
  - ChessNet:
  ```python
  # load the trained model:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  final_model_path = ...
  saved_model = ChessNet(hidden_layers=4, hidden_size=200)
  saved_model.load_state_dict(torch.load(final_model_path))
  saved_model.to(device)
  saved_model.eval()

  # make model predictions using choose_move func from utils
  board = chess.Board()
  # push moves onto the empty board
  curr_player = ... # chess.WHITE or chess.BLACK
  move = choose_move(board, curr_player, saved_model)
  board.push(move)
  display(board)
  ```
  - ChessEvalPosNet:
  ```python
  # load the trained model:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  saved_model = ChessPosEvalNet(hidden_layers=4, hidden_size=200)
  saved_model_path = ...
  saved_model.load_state_dict(torch.load(saved_model_path))
  saved_model.to(config.device)
  saved_model.eval()

  board = chess.Board()
  # push moves onto the empty board
  x = torch.tensor(board_to_rep(board))
  x = x.unsqueeze(0)
  x = x.to(device).float()
  output = saved_model(x)
  ```
### Using the Genetic Algorithm:
- Change the directory to `/GeneticAlgorithm`
- Running the Genetic Algorithm: 
  - Download the file lichess_elite_2022-12 from [Lichess](https://database.nikonoel.fr/) and place it in `/GeneticAlgorithm`
  - Run the bash file `/GeneticAlgorithm/GA_runner.sh`, which will run the genetic algorithm in the file `/GeneticAlgorithm/GA_training.py`. Run this file as many times as needed in order to create a population of genetically tuned organisms. 
- Running the Co-evolution Genetic Algorithm:
  - Run the bash file `/GeneticAlgorithm/coevolution_training.sh`, which will run the coevolution genetic algorithm in the file `/GeneticAlgorithm/coevolution_training.py`, resulting in the creation of a final genetically tuned organism. 
- Using the final organism in order to make predictions:
  ```python 
  # Example of using the final Organism Bit String to make predictions:
  final_optimal_organism = 'put your final bit string organism here'
  final_optimal_organism_parameters = bit_to_params(final_optimal_organism)
  board = chess.Board()
  depth = 3
  lambda_value = 0 # This value means we do not use the CNN
  get_best_move(board, depth, final_optimal_organism_parameters, lambda_value)
  ```

### Using the Lambda Bots:
- Change the directory to `/LambdaBots`

- Testing the Different Values of $\lambda$: 
  - Download the file lichess_elite_2022-01 from [Lichess](https://database.nikonoel.fr/) and place it in `/LambdaBots`
  - Run the file `/LambdaBots/lambda_external_test.ipynb` in order to compare how different values of $\lambda$ perform on an external dataset.  
  - Run the file `/LambdaBots/lambda_battle_arena.ipynb` in order to compare how different values of $\lambda$ perform when playing against each other. 
- Making two $\lambda$ bots play each other:
  ```python 
  # Example of a game between two bots that have different lambda values:
  final_optimal_organism = 'put your final bit string organism here'
  final_optimal_organism_parameters = bit_to_params(final_optimal_organism)
  white_parameters = final_optimal_organism_parameters
  black_parameters = final_optimal_organism_parameters
  depth = 3
  lambda_value_white = 0.2 
  lambda_value_black = 0.4
  runGame(depth, white_parameters, black_parameters, lambda_value_white, lambda_value_black)

  # In order to make the game of limited moves, or play from a specific position, use the following example: 
  max_num_moves = 20
  fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
  runGame_limited_moves(depth, white_parameters, black_parameters, max_num_moves, fen, lambda_value_white, lambda_value_black)
  ```


Note: the datasets used in the project need to be downloaded before certain files can be run. 
These datasets can be obtained from the following links: 

1. https://database.nikonoel.fr/

2. https://www.kaggle.com/datasets/arevel/chess-games


