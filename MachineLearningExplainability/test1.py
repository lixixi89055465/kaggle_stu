import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint_sequential

# a trivial model
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
    nn.ReLU()
)

# model input
input_var = torch.randn(1, 100, requires_grad=True)

# the number of segments to divide the model into
segments = 2

# finally, apply checkpointing to the model
# note the code that this replaces:
# out = model(input_var)
out = checkpoint_sequential(model , segments, input_var)

# backpropagate
out.sum().backwards()