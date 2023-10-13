# imports:
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from config import Config 
from data import ChessDataset
from complex_cnn.complex_model import ChessNet
from preprocess import preprocess_chess_data
from test import test_piece_accuracy

print("Started Hyperparamter Tuning the Complex Model")
config = Config()

# Device configuration
print(f'Using {config.device} to train')

# number of epochs each model will be trained for
num_epochs = 10

# datasets:
train_chess_data = preprocess_chess_data(config.train_data_dir)
data_train = ChessDataset(train_chess_data['AN'])
valid_chess_data = preprocess_chess_data(config.valid_data_dir)
data_valid = ChessDataset(valid_chess_data['AN'])
print("Loaded datasets")

# loss:
metric_from = nn.CrossEntropyLoss()
metric_to = nn.CrossEntropyLoss()

# hyperparam tuning using tensorboard:
batch_sizes = [32, 512, 2048]
learning_rates = [0.1, 0.001, 0.00001] 
num_models = len(batch_sizes) * len(learning_rates)

print(f"Tuning {num_models} models...")
for batch_size in batch_sizes: 
  data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
  data_valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, drop_last=True)
  n_total_steps = len(data_train_loader)
  for learning_rate in learning_rates:
    model = ChessNet(hidden_layers=4, hidden_size=200).to(config.device)  
    model.train()
    writer = SummaryWriter(f'{config.runs_dir}/hp_tuning_complex/BS_{batch_size}__LR_{learning_rate}')
    step = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # training loop:
    print()
    print(f'training model with batch size={batch_size}, LR={learning_rate}')
    for epoch in range(num_epochs):
      model.train()
      losses = []
      for i, batch in enumerate(data_train_loader):
        X, y, _ = batch
        X = X.to(config.device).float()
        y = y.to(config.device).float()

        # forward pass:
        output = model(X)

        # loss calculation:
        loss_from = metric_from(output[:,0,:], y[:,0,:])
        loss_to = metric_to(output[:,1,:], y[:,1,:])
        loss = loss_from + loss_to
        losses.append(loss)

        # backward and optimize:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # tensorboard logging (per epoch)
      avg_loss = sum(losses)/len(losses)
      print(f'avg loss for epoch {epoch+1} is {avg_loss}')
      writer.add_scalar('Training Loss', avg_loss, global_step=step)
      
      # Testing on valid set (per epoch)
      model.eval()
      print(f'epoch {epoch+1} accuracy on valid set:')
      from_accuracy, piece_accuracies, overall_piece_accuracy = test_piece_accuracy(model, data_valid_loader)
      print(f'from accuracy: {from_accuracy}')
      print(f'piece accuracy: [p,r,n,b,q,k]: {piece_accuracies}')
      print(f'overall piece accuracy: {overall_piece_accuracy}')
      writer.add_scalar('from_accuracy', from_accuracy, global_step=step)
      writer.add_scalar('overall_piece_accuracy', overall_piece_accuracy, global_step=step)

      step += 1
        
print(f'Finished tuning {num_models} models!')



