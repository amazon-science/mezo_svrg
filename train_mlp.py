import pytorch_lightning as pl
import torch 
import torch.nn as nn
from models.mlp_classification import MultiLayerPerceptron
from utils import MnistDataModule
from tqdm import tqdm
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_FO_mlp():
    mnist_dm = MnistDataModule(bs=64)
    mnistclassifier = MultiLayerPerceptron()

    if torch.cuda.is_available(): # if you have GPUs
        trainer = pl.Trainer(max_epochs=epochs, devices=1, accumulate_grad_batches=1, val_check_interval=0.1)
    else:
        trainer = pl.Trainer(max_epochs=epochs)

    trainer.fit(model=mnistclassifier, datamodule=mnist_dm)

    dict_results = {}
    dict_results['Tr_Loss'] = mnistclassifier.tr_loss
    dict_results['Time'] = mnistclassifier.time
    dict_results['Query'] = mnistclassifier.query
    with open('MNIST_FO_lr1e3_bs64.pickle', 'wb') as f:
        pickle.dump(dict_results, f)

def train_ZO_mlp():
    mnist_dm = MnistDataModule(bs=128)
    mnist_dm.setup(stage='fit')
    train_dataloader = mnist_dm.train_dataloader()
    val_dataloader = mnist_dm.val_dataloader()
    mnistclassifier = MultiLayerPerceptron(zero_order_eps=1e-3, learning_rate=1e-3)
    model = mnistclassifier.model
    #model.to(device)
    
    mnistclassifier.model.eval()
    
    for epoch in range(epochs):
        # validation loop
        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = mnistclassifier.validation_step_ZO(model, x, y)
        
        # training loop
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = mnistclassifier.training_step_ZO(model, (x, y))
    
    dict_results = {}
    dict_results['Tr_Loss'] = mnistclassifier.tr_loss
    dict_results['Time'] = mnistclassifier.time
    dict_results['Query'] = mnistclassifier.query
    with open('MNIST_ZO_lr1e3_bs128.pickle', 'wb') as f:
        pickle.dump(dict_results, f)
        
def train_ZO_SVRG_Coord_Rand_mlp():
    mnist_dm = MnistDataModule()
    mnist_dm.setup(stage='fit')
    train_dataloader = mnist_dm.train_dataloader()
    val_dataloader = mnist_dm.val_dataloader()
    mnistclassifier = MultiLayerPerceptron(zero_order_eps=1e-3, learning_rate=1e-3)
    model = mnistclassifier.model
    #model.to(device)
    
    mnistclassifier.model.eval()
    
    for epoch in range(epochs):
        print('epoch:', epoch)
        # validation loop
        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = mnistclassifier.validation_step_ZO(model, x, y)
        
        # training loop
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = mnistclassifier.training_step_ZO_SVRG_Rand_Coord(model, (x, y), epoch, i)
    dict_results = {}
    dict_results['Tr_Loss'] = mnistclassifier.tr_loss
    with open('MLP_ZO_SVRG_Coord_Rand_FD150_bs64_lr1e3.pickle', 'wb') as f:
        pickle.dump(dict_results, f)
        
def train_ZO_SVRG_mlp():
    mnist_dm = MnistDataModule(bs=9096)
    mnist_dm.setup(stage='fit')
    train_dataloader = mnist_dm.train_dataloader()
    val_dataloader = mnist_dm.val_dataloader()
    n_batches = len(train_dataloader)
    mnistclassifier = MultiLayerPerceptron(zero_order_eps=1e-3, learning_rate=1e-5, learning_rate_aux=2e-3, q=2)
    model = mnistclassifier.model
    #model.to(device)
    
    mnistclassifier.model.eval()
    
    for epoch in range(epochs):
        print('epoch:', epoch)
        # validation loop
        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = mnistclassifier.validation_step_ZO(model, x, y)
        
        # training loop
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            #x.to(device)
            #y.to(device)
            loss = mnistclassifier.training_step_ZO_SVRG(model, (x, y), epoch, i)
    dict_results = {}
    dict_results['Tr_Loss'] = mnistclassifier.tr_loss
    dict_results['Time'] = mnistclassifier.time
    dict_results['Query'] = mnistclassifier.query
    with open('MNIST_ZO_SVRG_q2_bs32_lr1e5.pickle', 'wb') as f:
        pickle.dump(dict_results, f)

    
if __name__ == "__main__":
    epochs = 100
    train_ZO_mlp()