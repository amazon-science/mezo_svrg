import pytorch_lightning as pl
import torch 
import torch.nn as nn
from models.least_squares import LeastSquares
from utils import MnistDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSDataModule(pl.LightningDataModule):
    def __init__(self, n=100, d=100):
        super().__init__()
        torch.manual_seed(0)
        self.X, self.w = torch.randn(n, d), torch.randn(d, 1)
        self.y = self.X@self.w + torch.randn(n, 1)*0.001
        LS_dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.LS_dataloader = DataLoader(LS_dataset, batch_size=32, shuffle=True)        

    def train_dataloader(self):
        return self.LS_dataloader

    def val_dataloader(self):
        return self.LS_dataloader

    def test_dataloader(self):
        return self.LS_dataloader

def train_FO_ls():
    LS_dm = LSDataModule(n, d)
    LS = LeastSquares(dim=d, learning_rate=1e-2)

    if torch.cuda.is_available(): # if you have GPUs
        trainer = pl.Trainer(max_epochs=epochs, devices=1, accumulate_grad_batches=1, val_check_interval=0.1)
    else:
        trainer = pl.Trainer(max_epochs=epochs)

    trainer.fit(model=LS, datamodule=LS_dm)
    
    dict_results = {}
    dict_results['Tr_Loss'] = LS.tr_loss
    dict_results['Time'] = LS.time
    dict_results['Query'] = LS.query
    with open('LS_SGD_n1000_d100_bs32_lr1e2.pickle', 'wb') as f:
        pickle.dump(dict_results, f)

def train_ZO_ls():
    LS_dm = LSDataModule(n, d)
    train_dataloader = LS_dm.train_dataloader()
    val_dataloader = LS_dm.val_dataloader()
    LS = LeastSquares(dim=d, learning_rate=2e-3, zero_order_eps=1e-3)
    model = LS.model
    #model.to(device)
    
    model.eval()
    
    for epoch in range(epochs):
        # validation loop
        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = LS.validation_step_ZO(model, x, y)
        
        # training loop
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = LS.training_step_ZO(model, (x, y))
    dict_results = {}
    dict_results['Tr_Loss'] = LS.tr_loss
    dict_results['Time'] = LS.time
    dict_results['Query'] = LS.query
    with open('LS_ZO_n1000_d100_bs64_lr2e3.pickle', 'wb') as f:
        pickle.dump(dict_results, f)
        
def train_ZO_SVRG_Coord_Rand_ls():
    LS_dm = LSDataModule(n, d)
    train_dataloader = LS_dm.train_dataloader()
    val_dataloader = LS_dm.val_dataloader()
    LS = LeastSquares(dim=d, learning_rate=2e-3, zero_order_eps=1e-3)
    model = LS.model
    #model.to(device)
    
    model.eval()
    
    for epoch in range(epochs):
        print('epoch:', epoch)
        # validation loop
        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = LS.validation_step_ZO(model, x, y)
        
        # training loop
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = LS.training_step_ZO_SVRG_Rand_Coord(model, (x, y), epoch, i)
    dict_results = {}
    dict_results['Tr_Loss'] = LS.tr_loss
    dict_results['Time'] = LS.time
    dict_results['Query'] = LS.query
    with open('LS_ZO_SVRG_Coord_Rand_FD_n1000_d100_bs64_lr2e3.pickle', 'wb') as f:
        pickle.dump(dict_results, f)
        
        
        
def train_ZO_SVRG_ls():
    LS_dm = LSDataModule(n, d)
    train_dataloader = LS_dm.train_dataloader()
    n_batches = len(train_dataloader)
    n_samples = len(train_dataloader.dataset)
    val_dataloader = LS_dm.val_dataloader()
    LS = LeastSquares(dim=d, q=2, learning_rate=1e-5, learning_rate_aux=1e-3, zero_order_eps=1e-3, n_samples=n_samples, n_batches=n_batches, X=LS_dm.X, y=LS_dm.y)
    model = LS.model
    #model.to(device)
    
    model.eval()
    
    for epoch in range(epochs):
        print('epoch:', epoch)
        # validation loop
        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = LS.validation_step_ZO(model, x, y)
        
        # training loop
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = LS.training_step_ZO_SVRG(model, (x, y), epoch, i)
    dict_results = {}
    dict_results['Tr_Loss'] = LS.tr_loss
    dict_results['Time'] = LS.time
    dict_results['Query'] = LS.query
    with open('LS_ZO_SVRG_q2_n1000_d100_lr1e3_bs64_full.pickle', 'wb') as f:
        pickle.dump(dict_results, f)
    
    
if __name__ == "__main__":
    n, d = 1000, 100
    epochs = 1000
    train_FO_ls()
    
    # X, y = np.random.randn(n, d), np.random.randn(n, 1)
    # w = np.linalg.inv(X.T@X)@X.T@y
    # print(np.sum((X@w - y)**2)/n)