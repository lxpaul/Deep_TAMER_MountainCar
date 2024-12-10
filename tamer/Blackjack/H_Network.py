import torch.nn as nn

class H_prediction_model(nn.Module):
    def __init__(self, input_dim):
        super(H_prediction_model,self).__init__()
        self.layer_1 = nn.Linear(input_dim,128)
        self.layer_2 = nn.Linear(128,64)
        self.layer_3 = nn.Linear(64,8)
        self.layer_4 = nn.Linear(8,1)

    def forward(self,x):
        x = nn.functional.relu(self.layer_1(x))
        x = nn.functional.relu(self.layer_2(x))
        x = nn.functional.relu(self.layer_3(x))
        x = nn.functional.tanh(self.layer_4(x))
        return x
    def __call__(self,x):
        return self.forward(x)