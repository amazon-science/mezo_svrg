from typing import Optional
from datetime import datetime
import time
import math
import numpy as np
import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from mezo_src.models import RobertaModelForPromptFinetuning
from mezo_src.modeling_roberta import RobertaConfig

#from memory_profiler import profile
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    RobertaForMaskedLM,
    OPTForSequenceClassification
)

class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        zero_order_eps: float = 1e-3,
        q: int = 1,
        learning_rate_aux: float = 1e-3,
        minibatch: int = 64,
        max_norm: float = 18000.0,
        z_std: float = 1.0,
        lr_anneal: float = 1.0,
        full_parameter: bool = True,
        batchsize_limit: int = 16,
        eval_splits: Optional[list] = None,
        soft_prompt: bool = False,
        use_SGD: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model_name = model_name_or_path
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.soft_prompt = soft_prompt

        if model_name_or_path == 'distilbert-base-cased' or model_name_or_path == 'roberta-large':
            if self.soft_prompt==True and model_name_or_path == 'roberta-large':
                config = RobertaConfig.from_pretrained(
                    'roberta-large',
                    num_labels=num_labels,
                    finetuning_task=self.hparams.task_name)
                self.model = RobertaModelForPromptFinetuning.from_pretrained(
                    "roberta-large",
                    config=config
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        elif 'gpt2' in model_name_or_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
            self.model.config.pad_token_id = self.model.config.eos_token_id
        elif 'opt' in  model_name_or_path:
            self.model = OPTForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        
        print(self.model)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        self.validation_step_outputs = []
        self.tr_loss = []
        self.tr_loss_minibatch = []
        self.time = []
        self.query = []
        self.grad_norm = []
        self.proj_val = []
        self.z_grad = []
        self.val_loss_ls = []
        self.val_acc = []
        self.lr_list = []
        self.memory_usage = []
        self.use_SGD = use_SGD
        
        self.zero_order_eps = zero_order_eps
        self.learning_rate = self.hparams.learning_rate
        self.q = q
        self.learning_rate_aux = learning_rate_aux
        self.minibatch = minibatch
        self.max_norm = max_norm
        self.z_std = z_std
        self.lr_anneal = lr_anneal
        self.full_parameter = full_parameter
        self.batchsize_limit = batchsize_limit

    def forward(self, **inputs):    
        return self.model(**inputs)
    
    def forward_ZO_val(self, inputs):
        model = self.model
        model.eval()
        batch_size = inputs['input_ids'].shape[0]
        iterations = math.ceil(batch_size/self.batchsize_limit)
        loss = 0
        acc = 0
        for i in range(iterations):
            input_batch = {}
            for k, v in inputs.items():
                input_batch[k] = v[i*self.batchsize_limit:min((i+1)*self.batchsize_limit, batch_size)]
            with torch.no_grad():
                outputs = model(**input_batch)
            loss += outputs[0].float()
            logits = outputs[1]
            if self.hparams.num_labels > 1:
                preds = torch.argmax(logits, axis=1)
            elif self.hparams.num_labels == 1:
                preds = logits.squeeze()
            
            labels = input_batch["labels"]
        
            # Compute Validation Accuracy
            correct_predictions = (preds == labels).sum().item()
            total_samples = len(labels)
            acc += correct_predictions / total_samples
        
        # freeing up memory    
        del input_batch, outputs, logits, preds
        return loss/iterations, acc/iterations 

    def training_step(self, batch, batch_idx):
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        outputs = self(**batch)
        loss = outputs[0]
        
        # for logging
        self.tr_loss.append(loss.detach().cpu().float().numpy())
        print('Minibatch Training Loss', loss)
        end_time = time.perf_counter()
        total_time = end_time-start_time
        print('Time taken (s): ', total_time)
        self.time.append(total_time)
        self.query.append(n)
        self.compute_grad_norm_fo(self.params_to_opt)        
        self.measure_memory_usage()
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):              
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        self.val_loss_ls.append(val_loss.detach().cpu().float().numpy())

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        val_info = {"loss": val_loss, "preds": preds, "labels": labels}

        # Compute Validation Accuracy
        correct_predictions = (preds == labels).sum().item()
        total_samples = len(labels)
        accuracy = correct_predictions / total_samples
        self.val_acc.append(accuracy)
        
        print('Validation Acc: ', accuracy)
        print('Validation Loss: ', val_loss)
        
        return val_info

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self.validation_step_outputs.clear()

    def validation_step_ZO(self, model, batch):
        val_loss, accuracy = self.forward_ZO_val(batch)
        self.val_loss_ls.append(val_loss.detach().cpu().float().numpy())

        self.val_acc.append(accuracy)
        
        print('Validation Acc: ', accuracy)
        print('Validation Loss: ', val_loss)


    def configure_params(self):
        if self.model_name == 'gpt2':
            layers = ['ln_f', 'h.11']
        elif self.model_name == 'gpt2-xl':
            layers = ['45', '46', '47']
        elif self.model_name == 'gpt2-medium':
            layers = ['22', '23']
        elif self.model_name == 'distilbert-base-cased':
            layers = ['classifier', 'layer.5']
        elif self.model_name == 'roberta-large':
            if self.soft_prompt==True:
                layers = ['classifier', 'lm_head', 'pooler', 'layer.23', 'layer.22', 'layer.21', 'layer.20'] 
            else:
                layers = ['classifier', 'layer.23', 'layer.22', 'layer.21', 'layer.20'] 

        model = self.model
        if self.full_parameter:
            self.params = [(n, p) for n, p in model.named_parameters()]
        else:    
            self.params = [(n, p) for n, p in model.named_parameters() if any(layer in n for layer in layers)]


    def configure_optimizers(self):
        """Prepare optimizer"""
        model = self.model

        for name, param in model.state_dict().items():
            print(name, param.size())
        
        #all_named_parameters = list(model.named_parameters())

        if self.model_name == 'gpt2':
            layers = ['ln_f', 'h.11']
        elif self.model_name == 'gpt2-xl':
            layers = ['45', '46', '47']
        elif self.model_name == 'gpt2-medium':
            layers = ['22', '23']
        elif self.model_name == 'distilbert-base-cased':
            layers = ['classifier', 'layer.5']
        elif self.model_name == 'roberta-large':
            if self.soft_prompt:
                layers = ['classifier', 'lm_head', 'pooler', 'layer.23', 'layer.22', 'layer.21', 'layer.20'] 
            else:
                layers = ['classifier', 'layer.23', 'layer.22', 'layer.21', 'layer.20'] 
        
        if self.full_parameter:
            self.params_to_opt = model.parameters()
        else: 
            self.params_to_opt = [p for n, p in model.named_parameters() if any(layer in n for layer in layers)]
        
        if self.use_SGD:
            # import pdb; pdb.set_trace()
            optimizer = torch.optim.SGD(self.params_to_opt, lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.params_to_opt, lr=self.hparams.learning_rate)
        return [optimizer]
    
    def zo_forward_memory_eff(self, model, inputs):
        model.eval()
        batch_size = inputs['input_ids'].shape[0]
        iterations = math.ceil(batch_size/self.batchsize_limit)
        loss = 0
        for i in range(iterations):
            input_batch = {}
            for k, v in inputs.items():
                input_batch[k] = v[i*self.batchsize_limit:min((i+1)*self.batchsize_limit, batch_size)]
            with torch.no_grad():
                outputs = model(**input_batch)
            loss += outputs[0].float()
        return loss/iterations    
    
    def zo_forward(self, model, inputs):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        return loss.detach()
    
    
    def efficient_perturb_parameters(self, parameters, random_seed: int, uniform: bool=False, use_beta: bool=False, scaling_factor=1):
        torch.manual_seed(random_seed)
        e = self.beta if use_beta else self.zero_order_eps
        for _, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * e
        return    
    
    
    def training_step_ZO(self, model, batch):
        # run ZO update
        n = batch['input_ids'].size(0)
        start_time = time.perf_counter()
        random_seed = np.random.randint(1000000000)
        parameters = self.params
        
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed)
            #if not self.half_precision:
            loss1 = self.zo_forward_memory_eff(model, batch)
            #else:
            #    loss1 = self.zo_forward(model, batch)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, scaling_factor=-2)
            #if not self.half_precision:
            loss2 = self.zo_forward_memory_eff(model, batch)
            #else:
            #    loss2 = self.zo_forward(model, batch)
        projected_grad = (loss1 - loss2) / (2 * self.zero_order_eps)
        
        model_dtype = next(self.model.parameters()).dtype
        projected_grad = projected_grad.to(model_dtype)
        
        # reset model back to its parameters at start of step
        self.efficient_perturb_parameters(parameters, random_seed)
        torch.manual_seed(random_seed)
        
        # compute SPSA gradient estimator
        for _, param in parameters:
            param.data = param.data - self.learning_rate * projected_grad * torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        
        # logging
        end_time = time.perf_counter()
        total_time = end_time-start_time
        print('Time taken (s): ', total_time)
        self.time.append(total_time)
        self.query.append(2*n)
        self.log_training_loss(model, batch, fullbatch=False)
        self.measure_memory_usage()
    
    def training_step_ZO_SVRG(self, model, batch, epoch, batch_idx, total_batches):
        # run ZO-SVRG update
        n = batch['input_ids'].size(0)
        print('Batch size: ', n)
        start_time = time.perf_counter()
        
        # current iteration
        curr_iteration = epoch * total_batches + batch_idx

        w = 2 * total_batches
        # updating spsa clipping
        if curr_iteration <= w:
            self.max_norm = self.max_norm if not self.grad_norm else max(self.max_norm, max(self.grad_norm)) 

        # learning rate scheduling strategy
        if len(self.tr_loss) > w:
            v1, v2 = self.get_average_np(self.tr_loss_minibatch, int(w/2))
            print('leading average: ', v1)
            print('trailing average: ', v2)
            if v1/v2 > 1.0:
                self.learning_rate_aux = max(self.learning_rate_aux/self.lr_anneal, 1e-4)
                self.learning_rate = max(self.learning_rate/self.lr_anneal, 1e-7)
            elif 1.0 >= v1/v2 > 0.999:
                self.learning_rate_aux = min(self.learning_rate_aux*self.lr_anneal/4, 5e-3)
                self.learning_rate = min(self.learning_rate*self.lr_anneal/4, 1e-5)
        print('Learning rate (full-batch): ', self.learning_rate_aux)
        print('Learning rate (mini-batch): ', self.learning_rate)
        self.lr_list.append(self.learning_rate_aux)
        
        # parameters contains tuples of params to optimize in list
        parameters = self.params
            
        # do full batch update every q steps
        if curr_iteration % self.q == 0:
            print('Full batch update')
            v = self.SPSA_estimator(model, parameters, batch)
            self.full_grad = self.clip_gradients_dict(v)
            self.parameters = parameters.copy()
            with torch.no_grad():
                for name, param in parameters:
                    param.data = param.data - self.learning_rate_aux * self.full_grad[name]
        else:
            # minibatch update
            print('Mini batch update')
            f_rand_curr = self.SPSA_estimator(model, parameters, batch)
            f_rand_past = self.SPSA_estimator(model, self.parameters, batch)
            v = {}
            with torch.no_grad():
                for name, param in parameters:
                    v[name] = f_rand_curr[name] - f_rand_past[name] + self.full_grad[name]
            v = self.clip_gradients_dict(v)
            with torch.no_grad():
                for name, param in parameters:
                    param.data = param.data - self.learning_rate * v[name]
        end_time = time.perf_counter()
        total_time = end_time-start_time
        print('Time taken (s): ', total_time)
        self.time.append(total_time)
        self.query.append(2*n)
        self.log_training_loss(model, batch, fullbatch=False)
        self.measure_memory_usage()


    def training_step_MeZO_SVRG(self, model, batch, epoch, batch_idx, total_batches):
        # run MeZO-SVRG update
        n = batch['input_ids'].size(0)
        print('Batch size: ', n)
        start_time = time.perf_counter()
        
        # current iteration
        curr_iteration = epoch * total_batches + batch_idx

        w = 2 * total_batches

        # learning rate scheduling strategy
        if len(self.tr_loss_minibatch) > 2*w and curr_iteration % total_batches == 0:
            v1, v2 = self.get_average_np(self.tr_loss_minibatch, int(w/2))
            print('leading average: ', v1)
            print('trailing average: ', v2)
            if v1/v2 > 1.05:
                self.learning_rate_aux = max(self.learning_rate_aux/self.lr_anneal, 1e-5)
                self.learning_rate = max(self.learning_rate/self.lr_anneal, 1e-6)
            # elif 1.0 >= v1/v2 > 0.999:
            #     self.learning_rate_aux = min(self.learning_rate_aux*self.lr_anneal/4, 5e-3)
            #     self.learning_rate = min(self.learning_rate*self.lr_anneal/4, 1e-5)
        print('Learning rate (full-batch): ', self.learning_rate_aux)
        print('Learning rate (mini-batch): ', self.learning_rate)
        self.lr_list.append(self.learning_rate_aux)
        
        # parameters contains tuples of params to optimize in list
        parameters = self.params
            
        # do full batch update every q steps
        if curr_iteration % self.q == 0:
            print('Full batch update')
            self.full_grad = self.SPSA_estimator(model, parameters, batch)
            self.parameters = parameters.copy()
            with torch.no_grad():
                for name, param in parameters:
                    param.data = param.data - self.learning_rate_aux * self.full_grad[name]
        else:
            # minibatch update
            print('Mini batch update')
            parameters = self.SPSA_estimator_me(model, parameters, batch, scale=-1) # in-place operation
            parameters = self.SPSA_estimator_me(model, self.parameters, batch) # in-place operation
            with torch.no_grad():
                for name, param in parameters:
                    param.data = param.data - self.learning_rate * self.full_grad[name]
        end_time = time.perf_counter()
        total_time = end_time-start_time
        print('Time taken (s): ', total_time)
        self.time.append(total_time)
        self.query.append(2*n)
        self.log_training_loss(model, batch, fullbatch=False)
        self.measure_memory_usage()


    def log_training_loss(self, model, batch, fullbatch=True):
        with torch.no_grad():
            # loss computation
            loss = self.zo_forward_memory_eff(model, batch)
        if fullbatch:
            print("Fullbatch Train Loss", loss)
            self.tr_loss.append(loss.detach().cpu().float().numpy())
        else:
            print("Minibatch Train Loss", loss)
            self.tr_loss_minibatch.append(loss.detach().cpu().float().numpy())
                        

    def central_difference_grad_est(self, model, parameters, batch, uniform=False):
        random_seed = np.random.randint(1000000000, size=1)
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
            #if not self.half_precision:
            loss1 = self.zo_forward_memory_eff(model, batch)
            #else:
            #    loss1 = self.zo_forward(model, batch)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-2)
            #if not self.half_precision:
            loss2 = self.zo_forward_memory_eff(model, batch)
            #else:
            #    loss2 = self.zo_forward(model, batch)
        proj_grad = (loss1 - loss2)/(2 * self.zero_order_eps)
        model_dtype = next(self.model.parameters()).dtype
        proj_grad = proj_grad.to(model_dtype)
        print(loss1 - loss2)
        print('Projected Grad: ', proj_grad)
        self.proj_val.append(torch.abs(proj_grad).detach().cpu().float().numpy())

        estimator = {}
        self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
        torch.manual_seed(random_seed)
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            estimator[name] = proj_grad*z
        return estimator
    
    def central_difference_grad_est_me(self, model, parameters, batch, scale, uniform=False):
        # memory efficient central difference spsa estimator
        random_seed = np.random.randint(1000000000, size=1)
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
            loss1 = self.zo_forward_memory_eff(model, batch)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-2)
            loss2 = self.zo_forward_memory_eff(model, batch)
        proj_grad = (loss1 - loss2)/(2 * self.zero_order_eps)
        #import pdb; pdb.set_trace()
        #print('Proj Gradient: ', proj_grad)
        self.proj_val.append(torch.abs(proj_grad).detach().cpu().float().numpy())

        self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
        torch.manual_seed(random_seed)
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scale * self.learning_rate * proj_grad * z
        return parameters    
    
    
    def forward_difference_grad_est(self, model, parameters, batch, uniform=False):
        random_seed = np.random.randint(1000000000, size=1)
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
            loss1 = self.zo_forward_memory_eff(model, batch)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-1)
            loss2 = self.zo_forward_memory_eff(model, batch)
        proj_grad = (loss1 - loss2)/self.zero_order_eps
        self.proj_val.append(torch.abs(proj_grad).detach().cpu().float().numpy())
        
        estimator = {}
        torch.manual_seed(random_seed)
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.randn(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=self.z_std, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            estimator[name] = proj_grad*z
        return estimator

    def SPSA_estimator(self, model, parameters, batch):
        return self.central_difference_grad_est(model, parameters, batch)
    
    def SPSA_estimator_me(self, model, parameters, batch, scale=1):
        # memory efficient SPSA estimator
        return self.central_difference_grad_est_me(model, parameters, batch, scale)
    
    def clip_gradients_dict(self, grad_dict):
        """Clip the gradients in a dictionary to a maximum norm."""
        total_norm = 0
        for param_name, grad in grad_dict.items():
            total_norm += grad.norm(2) ** 2
        total_norm = total_norm ** 0.5
        print('Norm:', total_norm)
        self.grad_norm.append(total_norm)
        clip_coef = self.max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for param_name, grad in grad_dict.items():
                grad_dict[param_name].mul_(clip_coef)

        return grad_dict
    
    def compute_grad_norm_zo(self, grad_dict):
        """Compute gradient norm."""
        total_norm = 0
        for param_name, grad in grad_dict.items():
            total_norm += grad.norm(2) ** 2
        total_norm = total_norm ** 0.5
        print('Gradient Norm:', total_norm)
        self.grad_norm.append(total_norm)
    
    def compute_grad_norm_fo(self, param):
        total_norm = 0
        for p in param:
            if not p.detach().grad is None:
                param_norm = p.detach().grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print('Grad Norm: ', total_norm)
        self.grad_norm.append(total_norm)
    
    def get_average_tensors(self, tensor_list, n=50):
        v1 = torch.stack(tensor_list[-n:]).mean(dim=0)
        v2 = torch.stack(tensor_list[-2*n:-n]).mean(dim=0)
        return v1, v2
    
    def get_average_np(self, np_list, n=50):
        l = np.array(np_list)
        v1 = np.mean(l[-n:])
        v2 = np.mean(l[-3*n:-2*n])
        return v1, v2
    
    def measure_memory_usage(self):
        device = next(self.model.parameters()).device
        allocated_memory_bytes = torch.cuda.memory_reserved(device)
        allocated_memory_gb = allocated_memory_bytes / (1024 ** 3)
        self.memory_usage.append(allocated_memory_gb)
        print('Memory usage (GB): ', allocated_memory_gb)