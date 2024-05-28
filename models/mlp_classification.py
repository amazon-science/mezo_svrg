import pytorch_lightning as pl
import torch 
import torch.nn as nn 
from torchmetrics import Accuracy
import numpy as np
import time
from torch.utils.data import DataLoader


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self,image_shape=(1, 28, 28), hidden_units=(32, 16), zero_order_eps=1e-3, learning_rate_aux=1e-3, learning_rate=1e-3, delta=1e-3, beta=1e-3, q=1, n_batches=1, train_dataloader=None):
        super().__init__()
        self.zero_order_eps = zero_order_eps
        self.learning_rate = learning_rate
        self.learning_rate_aux = learning_rate_aux
        self.FO = False
        self.tr_loss = []
        self.delta = delta
        self.beta = beta
        self.q = q
        self.n_batches = n_batches
        self.time = [0]
        self.query = []
        self.train_dataloader = train_dataloader 
        
        # new PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.valid_acc = Accuracy(task='multiclass', num_classes=10)
        self.test_acc = Accuracy(task='multiclass', num_classes=10)
        
        # Model similar to previous section:
        input_size = image_shape[0] * image_shape[1] * image_shape[2] 
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units: 
            layer = nn.Linear(input_size, hidden_unit) 
            all_layers.append(layer) 
            all_layers.append(nn.ReLU()) 
            input_size = hidden_unit 
 
        all_layers.append(nn.Linear(hidden_units[-1], 10)) 
        #all_layers.append(nn.Softmax(dim=1)) 
        self.model = nn.Sequential(*all_layers)
        self.d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def forward(self, x):
        x = self.model(x)
        return x
    
    def zo_forward(self, model, x, y):
        model.eval()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss.detach()
        
    def efficient_perturb_parameters(self, parameters, random_seed: int, uniform: bool=False, use_beta: bool=False, scaling_factor=1):
        torch.manual_seed(random_seed)
        e = self.beta if use_beta else self.zero_order_eps
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.rand(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * e
        return
    
    def efficient_perturb_parameters_scalar(self, parameter, use_beta: bool=False, scaling_factor=1):
        e = self.beta if use_beta else self.zero_order_eps
        return parameter + scaling_factor * e

    def central_difference_grad_est(self, model, parameters, x, y, uniform=False):
        random_seed = np.random.randint(1000000000, size=1)
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
            loss1 = self.zo_forward(model, x, y)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-2)
            loss2 = self.zo_forward(model, x, y)
        proj_grad = (loss1 - loss2)/(2 * self.zero_order_eps)
        
        estimator = {}
        self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
        torch.manual_seed(random_seed)
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.rand(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            estimator[name] = proj_grad*z
        return estimator
    
    def forward_difference_grad_est(self, model, parameters, x, y, uniform=False):
        random_seed = np.random.randint(1000000000, size=1)
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform)
            loss1 = self.zo_forward(model, x, y, reduction='mean')
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-1)
            loss2 = self.zo_forward(model, x, y, reduction='mean')
        proj_grad = (loss1 - loss2)/self.zero_order_eps
        
        estimator = {}
        torch.manual_seed(random_seed)
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.rand(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            estimator[name] = proj_grad*z
        return estimator


    def SPSA_estimator(self, model, parameters, x, y, fullbatch=True):
        if fullbatch:
            ds = self.train_dataloader.dataset
            dl = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=4)
            x, y = next(iter(dl))
        return self.central_difference_grad_est(model, parameters, x, y)

    
    def FD_estimator(self, model, parameters, x, y):
        estimator = {}
        for name, param in parameters:
            random_seed = np.random.randint(1000000000)
            p = param.data.view(-1, 1)
            r = torch.zeros_like(p)
            for i in range(p.size(0)):
                with torch.no_grad():
                    # first function evaluation
                    p[i] = self.efficient_perturb_parameters_scalar(p[i])
                    loss1 = self.zo_forward(model, x, y)
                    # second function evaluation
                    p[i] = self.efficient_perturb_parameters_scalar(p[i], scaling_factor=-2)
                    loss2 = self.zo_forward(model, x, y)
                    # reset model back to its parameters at start of step
                    p[i] = self.efficient_perturb_parameters_scalar(p[i])

                r[i] = (loss1 - loss2) / (2 * self.zero_order_eps)
            estimator[name] = r.view(param.data.size())
                
        return estimator
    
    def SPSA_estimator_per_sample(self, model, parameters, x, y):
        estimator = {}
        batch_size = x.shape[0]
        for i in range(batch_size):
            random_seed = np.random.randint(1000000000)
            with torch.no_grad():
                # first function evaluation
                self.efficient_perturb_parameters(parameters, random_seed, use_beta=True, uniform=True)
                loss1 = self.zo_forward(model, x[i], torch.unsqueeze(y[i], dim=0))
                # second function evaluation
                self.efficient_perturb_parameters(parameters, random_seed, scaling_factor=-1, use_beta=True, uniform=True)
                loss2 = self.zo_forward(model, x[i], torch.unsqueeze(y[i], dim=0))
            projected_grad = (loss1 - loss2) / self.beta

            torch.manual_seed(random_seed)
            for name, param in parameters:
                # uniform distribution over unit sphere
                z = torch.rand(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
                if not estimator or not name in estimator:
                    estimator[name] = projected_grad * z/batch_size
                else:
                    estimator[name] += projected_grad * z/batch_size
                
        return estimator


    def training_step_ZO_SVRG_Rand_Coord(self, model, batch, epoch, batch_idx):
        # run ZO-SVRG-Rand-Coord update
        x, y = batch
        start = time.time_ns()
        random_seed = np.random.randint(1000000000)
        parameters = []
        for name, param in model.named_parameters():
            parameters.append((name, param))
            
        if epoch % self.q == 0 and batch_idx % 40 == 0:
            n = self.d
            v = self.FD_estimator(model, parameters, x, y)
            #v = self.LS_true_grad(parameters, x, y)
            self.v_coord = v
            self.parameters = parameters.copy()
        else:
            n = x.size(0)
            f_rand_curr = self.SPSA_estimator_per_sample(model, parameters, x, y)
            f_rand_past = self.SPSA_estimator_per_sample(model, self.parameters, x, y)
            v = {}
            for name, _ in parameters:
                v[name] = f_rand_curr[name] - f_rand_past[name] + self.v_coord[name]
        
        for name, param in parameters:
            param.data = param.data - self.learning_rate * v[name]
        
        end = time.time_ns()
        self.time.append((end-start)*1e9)
        self.query.append(2*n)
        
        with torch.no_grad():
            # loss computation
            loss = self.zo_forward(model, x, y)
        print("train_loss", loss)
        self.tr_loss.append(loss.detach().cpu().numpy())
        return loss
    
    def training_step_ZO_SVRG(self, model, batch, epoch, batch_idx):
        # run ZO-SVRG update
        start = time.time_ns()
        x, y = batch
        parameters = []
        for name, param in model.named_parameters():
            parameters.append((name, param))
            
        if batch_idx % self.q == 0:
            n = x.size(0)
            self.full_grad = self.SPSA_estimator(model, parameters, x, y, fullbatch=False)
            with torch.no_grad():
                for name, param in parameters:
                    param.data = param.data - self.learning_rate_aux * self.full_grad[name]
            self.parameters = parameters.copy()
        else:
            bs = 32#x.size(0)
            n = bs
            f_rand_curr = self.SPSA_estimator(model, parameters, x[:bs], y[:bs], fullbatch=False)
            f_rand_past = self.SPSA_estimator(model, self.parameters, x[:bs], y[:bs], fullbatch=False)
            v = {}
            for name, param in parameters:
                v[name] = f_rand_curr[name] - f_rand_past[name] + self.full_grad[name]
                param.data = param.data - self.learning_rate * v[name]
        end = time.time_ns()
        self.time.append((end-start)*1e9)
        self.query.append(2*n)
        
        with torch.no_grad():
            # loss computation
            loss = self.zo_forward(model, x, y)
        print("train_loss", loss)
        self.tr_loss.append(loss.detach().cpu().numpy())
        return loss


    def training_step_ZO(self, model, batch):
        # run ZO update
        start = time.time_ns()
        x, y = batch
        n = x.size(0)
        random_seed = np.random.randint(1000000000)
        parameters = []
        for name, param in model.named_parameters():
            parameters.append((name, param))
        with torch.no_grad():
            # first function evaluation
            self.efficient_perturb_parameters(parameters, random_seed)
            loss1 = self.zo_forward(model, x, y)
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, scaling_factor=-2)
            loss2 = self.zo_forward(model, x, y)
        projected_grad = (loss1 - loss2) / (2 * self.zero_order_eps)
        
        # reset model back to its parameters at start of step
        self.efficient_perturb_parameters(parameters, random_seed)
        torch.manual_seed(random_seed)
        for name, param in parameters:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data - self.learning_rate * projected_grad * z
        end = time.time_ns()
        self.time.append((end-start)*1e9)
        self.query.append(2*n)
        
        with torch.no_grad():
            # loss computation
            loss = self.zo_forward(model, x, y)
        print("train_loss", loss)
        self.tr_loss.append(loss.detach().cpu().numpy())
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        n = x.size(0)
        start = time.time_ns()

        # Run FO update
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.tr_loss.append(loss.detach().cpu().numpy())
        end = time.time_ns()
        self.time.append((end-start)*1e9)
        self.query.append(n)

        return loss
                
    def validation_step_ZO(self, model, x, y):        
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        print("valid_loss", loss)
        return loss              
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid accuracy", self.valid_acc, prog_bar=True)
        self.log("valid_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
            

