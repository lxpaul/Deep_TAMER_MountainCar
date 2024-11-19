import torch.nn as nn

class H_prediction_model(nn.Module):
    def __init__(self, input_dim):
        super(H_prediction_model,self).__init__()
        self.layer_1 = nn.Linear(input_dim,16)
        self.layer_2 = nn.Linear(16,1)

    def forward(self,x):
        x = nn.functional.relu(self.layer_1(x))
        x = nn.functional.tanh(self.layer_2(x))
        return x
    def __call__(self,x):
        return self.forward(x)