# imports:
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from config import Config 
from data import EvalDataset
from complex_cnn.complex_model import ChessPosEvalNet
from preprocess import preprocess_eval_data

print("Started Training the CNN Board Evaluation Model")
config = Config()

# Device configuration
device = config.device
print(f'Using {device} to train')

num_epochs = 160

# datasets:
train_chess_data = preprocess_eval_data(f'{config.eval_dataset_dir}/train_dataset.csv')
data_train = EvalDataset(train_chess_data, length = 64*15)
valid_chess_data = preprocess_eval_data(f'{config.eval_dataset_dir}/valid_dataset.csv')
data_valid = EvalDataset(valid_chess_data, length = 32*8)
test_chess_data = preprocess_eval_data(f'{config.eval_dataset_dir}/test_dataset.csv')
data_test = EvalDataset(test_chess_data, length = 32*100)
print("Loaded datasets")

# loss:
criterion = nn.MSELoss()

data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, drop_last=True)
data_valid_loader = DataLoader(data_valid, batch_size=32, shuffle=False, drop_last=True)
data_test_loader = DataLoader(data_test, batch_size=32, shuffle=False, drop_last=True)

model = ChessPosEvalNet(hidden_layers=4, hidden_size=200).to(device)
model.train()
writer = SummaryWriter(f'{config.runs_dir}/PosEval')
step = 0
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training loop:
print(f'training model with batch size = 64, LR = 0.0001')
for epoch in range(num_epochs):   
    model.train()
    losses = []
    for i, batch in enumerate(data_train_loader):
        X, y = batch
        X = X.to(device).float()
        y = y.to(device).float()
        y = y.unsqueeze(1)

        # forward pass:
        output = model(X)

        # loss calculation:
        loss = criterion(output, y)
        losses.append(loss)

        # backward and optimize:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # tensorboard logging (per epoch)
    avg_loss = sum(losses)/len(losses)
    print(f'avg train loss for epoch {epoch+1} is {avg_loss}')
    writer.add_scalar('Training Loss', avg_loss, global_step=step)

    # Testing on valid set (per epoch)
    model.eval()
    losses = []
    for i, batch in enumerate(data_valid_loader):
        X, y = batch
        X = X.to(device).float()
        y = y.to(device).float()
        y = y.unsqueeze(1)

        # forward pass:
        output = model(X)

        # loss calculation:
        loss = criterion(output, y)
        losses.append(loss)
        
    # tensorboard logging (per epoch)
    avg_loss = sum(losses)/len(losses)
    print(f'avg valid loss of model at epoch {epoch+1} is {avg_loss}')
    writer.add_scalar('Valid Loss', avg_loss, global_step=step)

    print()
    step += 1
        
# Testing on test set
model.eval()
losses = []
for i, batch in enumerate(data_test_loader):
    X, y = batch
    X = X.to(device).float()
    y = y.to(device).float()
    y = y.unsqueeze(1)

    # forward pass:
    output = model(X)

    # loss calculation:
    loss = criterion(output, y)
    losses.append(loss)
        
# tensorboard logging (per epoch)
avg_loss = sum(losses)/len(losses)
print(f'Test MSE loss after {epoch+1} epochs of training is {avg_loss}')

print('Finished Training')
model_name = 'cnn_board_eval.pth'
PATH_name = f'{config.saved_models_dir}/{model_name}'
torch.save(model.state_dict(), PATH_name)
print(f'Saved model to {PATH_name}')

