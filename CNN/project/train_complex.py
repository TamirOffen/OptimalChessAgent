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

print("Started Training the Complex Model")
config = Config()

# Device configuration
device = config.device
print(f'Using {device} to train')

num_epochs = 75

# datasets:
train_chess_data = preprocess_chess_data(f'{config.chess_dataset_dir}/train_dataset.csv')
data_train = ChessDataset(train_chess_data['AN'])
valid_chess_data = preprocess_chess_data(f'{config.chess_dataset_dir}/valid_dataset.csv')
data_valid = ChessDataset(valid_chess_data['AN'], length=15_000)
test_chess_data = preprocess_chess_data(f'{config.chess_dataset_dir}/test_dataset.csv')
data_test = ChessDataset(test_chess_data['AN'])
print("Loaded datasets")

# loss:
metric_from = nn.CrossEntropyLoss()
metric_to = nn.CrossEntropyLoss()

data_train_loader = DataLoader(data_train, batch_size=config.batch_size, shuffle=True, drop_last=True)
data_valid_loader = DataLoader(data_valid, batch_size=config.batch_size, shuffle=False, drop_last=True)
data_test_loader = DataLoader(data_test, batch_size=config.batch_size, shuffle=False, drop_last=True)

model = ChessNet(hidden_layers=4, hidden_size=200).to(device)
model.train()
writer = SummaryWriter(f'{config.runs_dir}/final_model')
step = 0
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# training loop:
print(f'training model with batch size = {config.batch_size}, LR = {config.learning_rate}')
for epoch in range(num_epochs):
    model.train()
    losses = []
    for i, batch in enumerate(data_train_loader):
        X, y, _ = batch
        X = X.to(device).float()
        y = y.to(device).float()

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
    print(f'train: avg loss for epoch {epoch+1} is {avg_loss}')
    writer.add_scalar('Training Loss', avg_loss, global_step=step)
    
    # Testing on valid set (per epoch)
    model.eval()
    print(f'accuracy on valid set at end of training model ({epoch+1} epochs):')
    from_accuracy, piece_accuracies, overall_piece_accuracy = test_piece_accuracy(model, data_valid_loader)
    print(f'from accuracy: {from_accuracy}')
    print(f'piece accuracy: [p,r,n,b,q,k]: {piece_accuracies}')
    print(f'overall piece accuracy: {overall_piece_accuracy}')
    writer.add_scalar('from_accuracy', from_accuracy, global_step=step)
    writer.add_scalar('overall_piece_accuracy', overall_piece_accuracy, global_step=step)

    print()
    step += 1
        
# Testing on test set
model.eval()
print(f'accuracy on test set at end of training model ({num_epochs} epochs):')
from_accuracy, piece_accuracies, overall_piece_accuracy = test_piece_accuracy(model, data_test_loader)
print(f'from accuracy: {from_accuracy}')
print(f'piece accuracy: [p,r,n,b,q,k]: {piece_accuracies}')
print(f'overall piece accuracy: {overall_piece_accuracy}')

print('Finished Training')
model_name = 'cnn_final.pth'
PATH_name = f'{config.saved_models_dir}/{model_name}'
torch.save(model.state_dict(), PATH_name)
print(f'Saved model to {PATH_name}')

