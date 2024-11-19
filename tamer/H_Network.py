import torch as th
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self,input_dim):
        super(Encoder,self).__init__()
        self.conv_1 = nn.Conv1d(input_dim,8,2)
        self.conv_2 = nn.Conv1d(8,16,2)

        self.conv_bn_1 = nn.BatchNorm1d(8)
        self.conv_bn_2 = nn.BatchNorm1d(16)
        
        self.linear_1 = nn.Linear(16,100)

    def forward(self,x):
        x = nn.functional.max_pool1d(self.conv_bn_1(self.conv_1(x)))
        x = nn.functional.max_pool1d(self.conv_bn_2(self.conv_2(x)))
        
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        return x
    
    def __call__(self,x):
        return self.forward(x)
    
class H_prediction_model(nn.Module):
    def __init__(self):
        super(H_prediction_model,self).__init__()
        self.layer_1 = nn.Linear(100,16)
        self.layer_2 = nn.Linear(16,2)

    def forward(self,x):
        x = nn.functional.relu(self.layer_1(x))
        x = nn.functional.relu(self.layer_2(x))
        return x
    def __call__(self,x):
        return self.forward(x)