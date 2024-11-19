import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch as th

from H_Network import *

def main():
    H_model = H_prediction_model(3)

    optimizer = optim.Adam(H_model.parameters(),lr=0.01)
    loss_fn = nn.MSELoss()
    n_epochs = 10000

    DATA = np.load("Data/MountainCar_Data.npy")
    X_data = th.tensor(DATA[:,:3], dtype=th.float32)
    Y_data = th.tensor(DATA[:,3:], dtype=th.float32)
    for epoch in range(n_epochs):
        H_model.train()
        y_pred = H_model(X_data)
        loss = loss_fn(y_pred,Y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    th.save(H_model.state_dict(), "H_model_first_try")
if __name__ == '__main__':
    main()