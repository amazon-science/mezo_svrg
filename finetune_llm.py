import pickle
from tqdm import tqdm
import time
import torch
import argparse
import os
import numpy as np

from pytorch_lightning import Trainer
#from pytorch_lightning.strategies import DDPFullyShardedStrategy


from utils import GLUEDataModule, windowed_mean
from models.llm_module import GLUETransformer

def save_pickle(data, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to: {filepath}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Model Fine-tuning')

    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--samplesize', type=int, default=1024, help='Training data sample size')
    parser.add_argument('--samplesize_validation', type=int, default=128, help='Validation data sample size')
    parser.add_argument('--model_name', type=str, default='DistilBert', help='Name of the pre-trained model')
    parser.add_argument('--task', type=str, default='mnli', help='Task for model training')
    parser.add_argument('--full_parameter', action='store_true', help='True for full parameter fine-tuning')
    parser.add_argument('--algorithm', type=str, default='FO', help='Algorithm to use ("FO", "ZO", "ZOSVRG")')
    parser.add_argument('--q', type=int, default=2, help='q parameter used only for ZO-SVRG')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batchsize_limit', type=int, default=64, help='Max batch size to be used to avoid memory error')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Max sequence length for inputs')

    parser.add_argument('--anneal', type=float, default=1.5, help='Annealing parameter')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=int, default=0, help='GPU Number')
    parser.add_argument('--results', type=str, default='results_demo', help='Name of folder to store results')
    parser.add_argument('--lr_mezosvrg_mb', type=float, default=1e-6, help='Mini-batch learning rate for MeZO-SVRG')
    parser.add_argument('--perturbation_scale', type=float, default=1e-3, help='Perturbation scale for SPSA estimators')
    parser.add_argument('--soft_prompt', action='store_true', help='True for using soft prompt')
    parser.add_argument('--half_precision', action='store_true', help='Using half-precision fine-tuning')
    args = parser.parse_args()
    return args

def finetune_FO(device_num, algorithm, max_seq_length, model_name, task, samplesize, samplesize_validation, batchsize, batchsize_limit, lr, full_parameter, results_folder, soft_prompt, half_precision=False):
    dm = GLUEDataModule(
    model_name_or_path=model_name,
    task_name=task,
    max_seq_length=max_seq_length,
    sample_size=samplesize,
    train_batch_size=batchsize_limit,
    validation_sample_size=samplesize_validation,
    eval_batch_size=batchsize_limit,
    soft_prompt=soft_prompt
    )
    dm.setup("fit")
    # import pdb; pdb.set_trace()
    if 'SGD' in algorithm:
        use_SGD = True
    else:
        use_SGD = False
        
    transformer = GLUETransformer(
        model_name_or_path=model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=lr,
        full_parameter=full_parameter,
        soft_prompt=soft_prompt,
        use_SGD=use_SGD,
    )
    # perform gradient accumulation
    num_batches = int(batchsize/batchsize_limit)
    
    if half_precision:
        precision = 'bf16-true'
        trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=[device_num] if torch.cuda.is_available() else None,  # limiting got iPython runs
            precision=precision,
            accumulate_grad_batches = num_batches,
            #limit_val_batches=1,
            # devices=[0,1,2,3],
            # accelerator="gpu",
            # strategy="deepspeed",
        )
    else:
        trainer = Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=[device_num] if torch.cuda.is_available() else None,  # limiting got iPython runs
            accumulate_grad_batches = num_batches,
            #limit_val_batches=1,
            # devices=[0,1,2,3],
            # accelerator="gpu",
            # strategy="deepspeed",
        )
        

    
    start_time = time.time()
    trainer.fit(transformer, datamodule=dm)
    end_time = time.time()
    total_training_time = end_time-start_time
    
    dict_results = {}
    dict_results['Model'] = model_name
    dict_results['Task'] = task
    dict_results['BS'] = batchsize
    dict_results['LR'] = lr
    dict_results['Algorithm'] = 'FO-SGD'
    dict_results['Tr_Loss'] = transformer.tr_loss
    dict_results['Time'] = transformer.time
    dict_results['Query'] = transformer.query
    dict_results['Grad_Norm'] = transformer.grad_norm
    dict_results['Overall_Tr_Time'] = total_training_time
    dict_results['Val_Loss'] = transformer.val_loss_ls
    dict_results['Val_Acc'] = transformer.val_acc
    dict_results['Memory'] = transformer.memory_usage
 
    if 'facebook' in model_name:
        model_name = model_name.replace('facebook/', "")
    
    file_name = f'{model_name}_{task}_FO_lr{str(lr)}_bs{str(batchsize)}_samplesize{str(samplesize)}_fullparam{str(full_parameter)}.pickle'
    save_pickle(dict_results, results_folder, file_name)
    
    print('Finished Task ' + task + ' with full parameter being ' + str(full_parameter))
    print('-----------------Statistics-----------------')
    window_size_tr = int(np.ceil(len(transformer.tr_loss) / epochs))
    arr_tr_loss = windowed_mean(transformer.tr_loss, window_size_tr)
    print('Best Training Loss: ', np.nanmin(arr_tr_loss))
    window_size_val = 2#int(np.ceil(len(transformer.val_acc) / epochs))
    arr_val_acc = windowed_mean(transformer.val_acc, window_size_val)
    print('Best Validation Accuracy: ', np.max(arr_val_acc))
    print('Peak Memory Usage (GB): ', np.max(transformer.memory_usage))
    print('Total queries: ', np.sum(transformer.query)) 
       
def finetune_FO_warmup(transformer, dm, warmup_epochs):
    dm = GLUEDataModule(
    model_name_or_path="distilbert-base-cased",
    task_name="mnli",
    sample_size=1024,
    train_batch_size=32,
    eval_batch_size=64
    )
    trainer = Trainer(
        max_epochs=warmup_epochs,
        accelerator="auto",
        devices=[device_num] if torch.cuda.is_available() else None,  # limiting got iPython runs
        limit_val_batches=1,
    )
    trainer.fit(transformer, datamodule=dm)
    

def finetune_ZO(device, max_seq_length, model_name, task, samplesize, samplesize_validation, batchsize, batchsize_limit, lr, full_parameter, results_folder, perturbation_scale=1e-3, soft_prompt=False, half_precision=False):
    # Initializing Data Module
    dm = GLUEDataModule(
    model_name_or_path=model_name,
    task_name=task,
    max_seq_length=max_seq_length,
    sample_size=samplesize,
    train_batch_size=batchsize,
    validation_sample_size=samplesize_validation,
    eval_batch_size=batchsize,
    soft_prompt=soft_prompt
    )
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    # train_full_dataloader = dm.train_full_dataloader()
    val_dataloader = dm.val_dataloader()
    
    print('Memory Datamodule: ', torch.cuda.memory_reserved())
    
    # Initializing Transformer
    transformer = GLUETransformer(
        model_name_or_path=model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=lr,
        full_parameter=full_parameter,
        batchsize_limit = batchsize_limit,
        zero_order_eps=perturbation_scale,
        soft_prompt=soft_prompt
    )
    if half_precision:
        transformer.to(torch.bfloat16)
    model = transformer.model
    model.to(device)
    transformer.configure_params()
    
    print('Memory Transformer: ', torch.cuda.memory_reserved())

    
    transformer.model.eval()
    start_time = time.time()
    
    for epoch in range(epochs):
        print('ZO, Epoch', epoch)
        # validation loop
        for _, batch in enumerate(tqdm(val_dataloader)):
            print('Validation Loop')
            b = {}
            for k, v in batch.items():
                b[k] = v.to(device)
            # print('Memory After loading batch: ', torch.cuda.memory_reserved())
            transformer.validation_step_ZO(model, b)
            # print('Memory After Validation: ', torch.cuda.memory_reserved())
            # break

        
        # training loop
        for _, batch in enumerate(tqdm(train_dataloader)):
            print('ZO, Epoch', epoch)
            b = {}
            for k, v in batch.items():
                b[k] = v.to(device)
            # print('Memory After loading training batch: ', torch.cuda.memory_reserved())
            transformer.training_step_ZO(model, b)
            # print('Memory After Training: ', torch.cuda.memory_reserved())
            # break
    
    end_time = time.time()
    total_training_time = end_time-start_time    
    
    dict_results = {}
    dict_results['Model'] = model_name
    dict_results['Task'] = task
    dict_results['BS'] = batchsize
    dict_results['LR'] = lr
    dict_results['Algorithm'] = 'MeZO'
    dict_results['Tr_Loss'] = transformer.tr_loss_minibatch
    dict_results['Time'] = transformer.time
    dict_results['Query'] = transformer.query
    dict_results['Grad_Norm'] = transformer.grad_norm
    dict_results['Overall_Tr_Time'] = total_training_time
    dict_results['Val_Loss'] = transformer.val_loss_ls
    dict_results['Val_Acc'] = transformer.val_acc
    dict_results['Memory'] = transformer.memory_usage
 
    if 'facebook' in model_name:
        model_name = model_name.replace('facebook/', "")
        
    file_name = f'{model_name}_{task}_ZO_lr{str(lr)}_bs{str(batchsize)}_samplesize{str(samplesize)}_fullparam{str(full_parameter)}_perturbationscale{str(perturbation_scale)}.pickle'
    save_pickle(dict_results, results_folder, file_name)
        
    print('Finished Task ' + task + ' with full parameter being ' + str(full_parameter))
    print('-----------------Statistics-----------------')
    window_size_tr = int(np.ceil(len(transformer.tr_loss_minibatch) / epochs))
    arr_tr_loss = windowed_mean(transformer.tr_loss_minibatch, window_size_tr)
    print('Best Training Loss: ', np.nanmin(arr_tr_loss))
    window_size_val = int(np.ceil(len(transformer.val_acc) / epochs))
    arr_val_acc = windowed_mean(transformer.val_acc, window_size_val)
    print('Best Validation Accuracy: ', np.max(arr_val_acc))
    print('Peak Memory Usage (GB): ', np.max(transformer.memory_usage))
    print('Total queries: ', np.sum(transformer.query))

def finetune_ZO_SVRG(device, max_seq_length, model_name, task, samplesize, samplesize_validation, batchsize, batchsize_limit, lr_fullbatch, full_parameter, results_folder, lr_minibatch=1e-6, q=1, anneal=5, random_permute=True, perturbation_scale=1e-3, soft_prompt=False, half_precision=False):
    # Initializing Data Module
    
    # full-batch dataloader
    dm = GLUEDataModule(
    model_name_or_path=model_name,
    task_name=task,
    max_seq_length=max_seq_length,
    sample_size=samplesize,
    train_batch_size=batchsize,
    validation_sample_size=samplesize_validation,
    eval_batch_size=batchsize,
    soft_prompt=soft_prompt
    )
    dm.setup("fit")
    train_dataloader = dm.train_full_dataloader()
    train_mb_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    
    # Initializing Transformer
    transformer = GLUETransformer(
        model_name_or_path=model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=lr_minibatch,
        q=q,
        learning_rate_aux=lr_fullbatch,
        lr_anneal = anneal,
        full_parameter = full_parameter,
        batchsize_limit = batchsize_limit,
        zero_order_eps=perturbation_scale,
        soft_prompt=soft_prompt
    )
    if half_precision:
        transformer.to(torch.bfloat16)
    transformer.configure_params()
    model = transformer.model
    model.to(device)
    if random_permute:
        total_batches = len(train_mb_dataloader)
    else:    
        total_batches = len(train_dataloader)
    transformer.model.eval()
    start_time = time.time()
    
    for epoch in range(epochs):
        print('ZO-SVRG, Epoch', epoch)
        
        # validation loop
        for i, batch in enumerate(tqdm(val_dataloader)):
            print('Validation Loop')
            b = {}
            for k, v in batch.items():
                b[k] = v.to(device)
            transformer.validation_step_ZO(model, b)

        
        # training loop - random permutation
        if random_permute:
            print('Minibatch sampling using random permutation') 
            for i, batch in enumerate(tqdm(train_mb_dataloader)):
                print('ZO-SVRG, Epoch', epoch)
                # get full batch every q steps
                curr_iteration = epoch * total_batches + i
                if curr_iteration % q == 0:
                    print('Full-Batch Iteration')
                    batch = next(iter(train_dataloader))
                
                b = {}
                for k, v in batch.items(): 
                    b[k] = v.to(device)
                    
                    
                
                transformer.training_step_MeZO_SVRG(model, b, epoch, i, total_batches)
                
            
        # training loop - random sampling
        else:
            print('Minibatch sampling using random sampling') 
            for i, batch in enumerate(tqdm(train_dataloader)):
                # get full batch every q steps
                curr_iteration = epoch * total_batches + i
                
                b = {}
                for k, v in batch.items():
                    if curr_iteration % q == 0:
                        print('Full batch') 
                        b[k] = v.to(device)
                    else:
                        print('Mini batch')
                        b[k] = v[:batchsize].to(device)
                    
                
                transformer.training_step_MeZO_SVRG(model, b, epoch, i, total_batches)
                
    
    end_time = time.time()
    total_training_time = end_time-start_time 
    
    dict_results = {}
    dict_results['Model'] = model_name
    dict_results['Task'] = task
    dict_results['BS'] = batchsize
    dict_results['LR'] = lr_fullbatch
    dict_results['Algorithm'] = 'ZO-SVRG'
    dict_results['Tr_Loss'] = transformer.tr_loss_minibatch
    dict_results['Tr_Loss_Fullbatch'] = transformer.tr_loss
    dict_results['Time'] = transformer.time
    dict_results['Query'] = transformer.query
    dict_results['Grad_Norm'] = transformer.grad_norm
    dict_results['Proj_Val'] = transformer.proj_val
    dict_results['Overall_Tr_Time'] = total_training_time
    dict_results['Val_Loss'] = transformer.val_loss_ls
    dict_results['Val_Acc'] = transformer.val_acc
    dict_results['LR_List'] = transformer.lr_list
    dict_results['Memory'] = transformer.memory_usage
    
    if 'facebook' in model_name:
        model_name = model_name.replace('facebook/', "")
    
    file_name = f'{model_name}_{task}_ZO_SVRG_q{str(q)}_lr{str(lr)}_bs{str(batchsize)}_samplesize{str(samplesize)}_fullparam{str(full_parameter)}_anneal{str(anneal)}_perturbationscale{str(perturbation_scale)}.pickle'    
    save_pickle(dict_results, results_folder, file_name)
    
        
    print('Finished Task ' + task + ' with full parameter being ' + str(full_parameter))
    print('-----------------Statistics-----------------')
    window_size_tr = int(np.ceil(len(transformer.tr_loss_minibatch) / epochs))
    arr_tr_loss = windowed_mean(transformer.tr_loss_minibatch, window_size_tr)
    print('Best Training Loss: ', np.nanmin(arr_tr_loss))
    window_size_val = int(np.ceil(len(transformer.val_acc) / epochs))
    arr_val_acc = windowed_mean(transformer.val_acc, window_size_val)
    print('Best Validation Accuracy: ', np.max(arr_val_acc))
    print('Peak Memory Usage (GB): ', np.max(transformer.memory_usage))
    print('Total queries: ', np.sum(transformer.query))

def finetune_ZO_SVRG_with_warmup():
    # Initializing Data Module
    dm = GLUEDataModule(
    model_name_or_path="distilbert-base-cased",
    task_name="mnli",
    sample_size=1024,
    train_batch_size=1024,
    eval_batch_size=64,
    )
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    
    # Initializing Transformer
    transformer = GLUETransformer(
        model_name_or_path="distilbert-base-cased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=1e-2,
        q=1,
        learning_rate_aux=5e-3,
        max_norm=2.0,
    )
    
    warmup_epochs = 10
    finetune_FO_warmup(transformer, dm, warmup_epochs)
    
    
    model = transformer.model
    model.to(device)
    
    transformer.model.eval()
    
    for epoch in range(epochs):
        print('Epoch', epoch)
        # validation loop
        # for i, batch in enumerate(tqdm(val_dataloader)):
        #     #x.to(device)
        #     #y.to(device)
        #     loss = transformer.validation_step_ZO(model, batch)
        
        # training loop
        for i, batch in enumerate(tqdm(train_dataloader)):
            b = {}
            for k, v in batch.items():
                b[k] = v.to(device)
            #y.to(device)
            #batch.to(device)
            loss = transformer.training_step_ZO_SVRG(model, b, epoch, i)
    
    dict_results = {}
    dict_results['Tr_Loss'] = transformer.tr_loss
    dict_results['Time'] = transformer.time
    dict_results['Query'] = transformer.query
    dict_results['Grad_Norm'] = transformer.grad_norm
    with open('MNLI_ZO_SVRG_q1_lr5e3_withclip_maxnorm20_full_warmup.pickle', 'wb') as f:
        pickle.dump(dict_results, f)



if __name__ == "__main__":
    args = parse_arguments()
    
    # fine-tuning setup
    epochs = args.epochs
    samplesize = args.samplesize
    samplesize_validation = args.samplesize_validation
    
    # define model name
    model_name = args.model_name
    
    task = args.task
    full_parameter = args.full_parameter
    algorithm = args.algorithm
    device_num = args.device
    device = torch.device("cuda:" + str(device_num))
    results_folder = args.results
    
    # algorithm hyperparameters
    q = args.q
    batchsize = args.batchsize
    batchsize_limit = args.batchsize_limit
    anneal = args.anneal
    lr = args.lr
    lr_mezosvrg_mb = args.lr_mezosvrg_mb
    max_seq_length = args.max_seq_length
    perturbation_scale = args.perturbation_scale
    soft_prompt = args.soft_prompt
    half_precision = args.half_precision
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'Fine-tuning {model_name} on {task} with a dataset of size {str(samplesize)} and validation dataset of size {str(samplesize_validation)}.')
    print(f"GPU Nr: {args.device}")
    print(f"Max Seq Length (input): {args.max_seq_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Full Parameter: {args.full_parameter}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Q: {args.q}")
    print(f"Batch Size: {args.batchsize}")
    print(f"Batch Size: {args.batchsize_limit}")
    print(f"Anneal: {args.anneal}")
    print(f"Learning Rate: {args.lr}")
    print(f"Mini-batch Learning Rate (MeZO-SVRG only): {args.lr_mezosvrg_mb}")
    print(f"Perturbation Scale: {args.perturbation_scale}")
    print(f"Prompt Setting: {args.soft_prompt}")
    print(f"Half Precision: {args.half_precision}")
    print(f"Results folder: {args.results}")
    
    if 'FO' in algorithm:
        finetune_FO(device_num, algorithm, max_seq_length, model_name, task, samplesize, samplesize_validation, batchsize, batchsize_limit, lr, full_parameter, results_folder, soft_prompt, half_precision)
    elif algorithm == 'ZO':
        finetune_ZO(device, max_seq_length, model_name, task, samplesize, samplesize_validation, batchsize, batchsize_limit, lr, full_parameter, results_folder, perturbation_scale=perturbation_scale, soft_prompt=soft_prompt, half_precision=half_precision)
    else:
        finetune_ZO_SVRG(device, max_seq_length, model_name, task, samplesize, samplesize_validation, batchsize, batchsize_limit, lr, full_parameter, results_folder, lr_minibatch=lr_mezosvrg_mb, q=q, anneal=anneal, perturbation_scale=perturbation_scale, soft_prompt=soft_prompt, half_precision=half_precision)    