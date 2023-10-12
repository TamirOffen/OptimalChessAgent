# AI Project - CS236502
Spring 2023

| Student | Student ID | Email | 
| ---     | --- | --- |
| Aron Klevansky| 941190845 | klevansky@campus.technion.ac.il |
| Niv Ostroff | 212732101 | nivostroff@campus.technion.ac.il |
| Tamir Offen | 211621479 | tamiroffen@campus.technion.ac.il |

## Research Question: 

> What is the optimal tradeoff between genetically tuned handcrafted features vs. neural trained features when creating a chess playing agent?


Summary of Project: TODO Aron

---

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
- Chapter 3.2 - TODO Aron

### `/LambdaBots`
- Chapters 4 and 5 ? - TODO Aron



---

## Getting Started

TODO Tamir

To get started with the project, follow these steps:

1. Clone this repository: `git clone https://github.com/your-username/your-repo.git`
2. Navigate to the project directory: `cd your-repo`
3. Set up the environment by installing the required packages: `pip install -r requirements.txt`
4. Explore the `/notebooks` directory for examples and tutorials on how to use the chess-playing agent.
5. Use the code in the `/src` directory to integrate the agent into your own projects.


