# AI Project - CS236502
Spring 2023

| Student | Student ID | Email | 
| ---     | --- | --- |
| Aron Klevansky| 941190845 | klevansky@campus.technion.ac.il |
| Niv Ostroff | 212732101 | nivostroff@campus.technion.ac.il |
| Tamir Offen | 211621479 | tamiroffen@campus.technion.ac.il |

## Research Question: 

> What is the optimal tradeoff between genetically tuned handcrafted features vs. neural trained features when creating a chess playing agent?


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

### Final Words:
We hope that our work can inspire future chess engine builders, from mere hobbyists to those professionally dedicated to the field. Thank you for looking at our project! 

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
  - `/CNN/project/saved_models`: saved pytroch models for ChessPosEvalNet and ChessNet.
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

## Getting Started

To get started with the project, follow these steps:

1. Clone this repository: `git clone https://github.com/your-username/your-repo.git`
2. Navigate to the project directory: `cd your-repo`
3. Set up the environment by installing the required packages: `pip install -r requirements.txt`
4. Explore the `/notebooks` directory for examples and tutorials on how to use the chess-playing agent.
5. Use the code in the `/src` directory to integrate the agent into your own projects.

Note: the datasets used in the project need to be downloaded before certain files can be run. 
These datasets can be obtained from the following links: 

1. https://database.nikonoel.fr/

2. https://www.kaggle.com/datasets/arevel/chess-games


