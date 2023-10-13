"""
CNN Architecture:

Define your CNN architecture in a separate module (e.g., model.py).
Encapsulate your neural network layers, forward pass, and any custom layers you may have.
This module should allow you to easily create and modify your model.

note: separate model file for each model
"""


# imports:
import torch
import torch.nn as nn
import torch.nn.functional as F


# credit: Moran Reznik
class module(nn.Module):
  def __init__(self, hidden_size):
    super(module, self).__init__()
    # layers:
    # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    # nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(hidden_size)
    self.bn2 = nn.BatchNorm2d(hidden_size)
    self.activation1 = nn.SELU()
    self.activation2 = nn.SELU()

  def forward(self, x):
    x_input = torch.clone(x) # for residual connection
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.activation1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = x + x_input # residual connection
    x = self.activation2(x)
    return x
      

"""
default model:
ChessNet(
  (input_layer): Conv2d(6, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (module_list): ModuleList(
    (0-3): 4 x module(
      (conv1): Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn2): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation1): SELU()
      (activation2): SELU()
    )
  )
  (output_layer): Conv2d(200, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
"""
# credit: Moran Reznik
class ChessNet(nn.Module):
  def __init__(self, hidden_layers=4, hidden_size=200):
    super(ChessNet, self).__init__()
    self.hidden_layers = hidden_layers
    # we input a 6-layer rep of a board: layer for each type of piece
    # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)

    # need to use nn.ModuleList and not a normal python list
    self.module_list = nn.ModuleList([module(hidden_size) for _ in range(hidden_layers)])

    # output to and from matrix (which is why our out_channels = 2)
    self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride = 1, padding=1)

  def forward(self, x):
    x = self.input_layer(x)
    x = F.relu(x)
    for i in range(self.hidden_layers):
      x = self.module_list[i](x)
    x = self.output_layer(x)
    return x


"""
default model:
ChessPosEvalNet(
  (input_layer): Conv2d(6, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (module_list): ModuleList(
    (0-3): 4 x module(
      (conv1): Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn2): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation1): SELU()
      (activation2): SELU()
    )
  )
  (fc1): Linear(in_features=12800, out_features=6400, bias=True)
  (fc2): Linear(in_features=6400, out_features=1, bias=True)
)
"""
class ChessPosEvalNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessPosEvalNet, self).__init__()
        self.hidden_layers = hidden_layers
        
        # input layer
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
    
        # hidden layers
        self.module_list = nn.ModuleList([module(hidden_size) for _ in range(hidden_layers)])
    
        # fc head
        self.fc1 = nn.Linear(8*8*hidden_size, 8*8*hidden_size // 2)
        self.fc2 = nn.Linear(8*8*hidden_size // 2, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for i in range(self.hidden_layers):
          x = self.module_list[i](x)
        x = x.view(x.size(0), -1)  # flattens tensor, but keeps batch size
        x = self.fc1(x)
        x = self.fc2(x)
        return x



