from data import make_synth_data
from model import Model
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

def train(epochs:int,batch_size:int,sample:int,feature:int):

    x_dataset, y_dataset = make_synth_data(sample,feature)
    assert len(x_dataset) == len(y_dataset)


    model = Model(x_dataset.size(1),20,y_dataset.size(1))
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch in tqdm(range(epochs),leave=True):
        batch_loss = []
        for batch_ind in tqdm(range(0,len(x_dataset),batch_size),leave=False):
            start_ind = batch_ind
            end_ind = batch_ind + batch_size
            x_batch, y_batch = x_dataset[start_ind:end_ind,], y_dataset[start_ind:end_ind]    
            optim.zero_grad()
            y_batch_pred = model(x_batch)
            loss = loss_fn(y_batch_pred,y_batch)
            loss.backward()
            optim.step()
            batch_loss.append(loss.item())
        print(f"Epoch Loss {epoch}/{epochs}: ", sum(batch_loss)/len(batch_loss))



if __name__ == "__main__":
    train(10,4,5000,2)