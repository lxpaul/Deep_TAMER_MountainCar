import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch as th
import os as os
import asyncio

from H_Network import *

async def main(name_model:str,name_data:str):
    H_model = H_prediction_model(4)
    optimizer = optim.Adam(H_model.parameters(),lr=0.01)
    loss_fn = nn.MSELoss()
    n_epochs = 10000

    DATA = np.load(os.path.join("tamer/Blackjack/Data",name_data))
    X_data = th.tensor(DATA[:,:-1], dtype=th.float32)
    Y_data = th.tensor(DATA[:,-1:], dtype=th.float32)
    for epoch in range(n_epochs):
        H_model.train()
        y_pred = H_model(X_data)
        loss = loss_fn(y_pred,Y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    th.save(H_model.state_dict(), os.path.join("tamer/Blackjack/saved_models",name_model))

if __name__ == '__main__':
    name_data = input("name of the data set: ")
    name_model = input("name of the model: ")
    asyncio.run(main(name_model=name_model,name_data=name_data))