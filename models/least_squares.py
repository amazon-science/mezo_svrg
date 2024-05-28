import pytorch_lightning as pl
import torch 
import torch.nn as nn 
from torchmetrics import Accuracy
import numpy as np
import time


class LeastSquares(pl.LightningModule):
    def __init__(self, X=None, y=None, dim=100, zero_order_eps=1e-3, learning_rate_aux=1e-3, learning_rate=1e-3, delta=1e-3, beta=1e-3, q=1, n_samples=1, n_batches=1):
        super().__init__()
        self.zero_order_eps = zero_order_eps
        self.learning_rate = learning_rate
        self.learning_rate_aux = learning_rate_aux
        self.FO = False
        self.tr_loss = []
        self.grad_dot = []
        self.abs_proj = []
        self.time = [0]
        self.query = [] 
        self.delta = delta
        self.beta = beta
        self.q = q
        self.n_batches = n_batches
        self.n_samples = n_samples
        print(self.n_batches)
        self.X = X
        self.y = y

        
        all_layers = [nn.Linear(dim, 1, bias=False)] 
        self.model = nn.Sequential(*all_layers)
        self.d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def zo_forward(self, model, x, y, reduction='mean'):
        model.eval()
        pred = model(x)
        if reduction=='mean':
            loss = torch.mean((pred - y)**2)
        else:
            loss = torch.sum((pred - y)**2)
        return loss.detach()
    
    def efficient_perturb_parameters(self, parameters, random_seed: int, uniform: bool=False, use_beta: bool=False, scaling_factor=1):
        torch.manual_seed(random_seed)
        e = self.beta if use_beta else self.zero_order_eps
        for name, param in parameters:
            if uniform:
                # uniform distribution over unit sphere
                z = torch.rand(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
                #print('z', z)
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
            loss1 = self.zo_forward(model, x, y, reduction='mean')
            # second function evaluation
            self.efficient_perturb_parameters(parameters, random_seed, uniform=uniform, scaling_factor=-2)
            loss2 = self.zo_forward(model, x, y, reduction='mean')
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
    
    def SPSA_estimator_sample(self, model, parameters, x, y, fullbatch=True):
        estimator = {}
        for name, _ in parameters:
            estimator[name] = 0
        if fullbatch:
            n = self.X.shape[0]
            x, y = self.X, self.y
        else:
            n = x.size(0)
        for i in range(n):
            grad_hat = self.forward_difference_grad_est(model, parameters, x[i], y[i])
            for name, _ in parameters:
                estimator[name] += grad_hat[name]/n
        return estimator
    
    def SPSA_estimator(self, model, parameters, x, y, fullbatch=True):
        if fullbatch:
            n = self.X.shape[0]
            x, y = self.X, self.y
        else:
            n = x.size(0)
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
                loss1 = self.zo_forward(model, x[i], y[i])
                # second function evaluation
                self.efficient_perturb_parameters(parameters, random_seed, scaling_factor=-1, use_beta=True, uniform=True)
                loss2 = self.zo_forward(model, x[i], y[i])
            projected_grad = (loss1 - loss2) / self.beta

            torch.manual_seed(random_seed)
            for name, param in parameters:
                # uniform distribution over unit sphere
                z = torch.rand(size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                z = z / torch.linalg.norm(z)
                if not estimator:
                    estimator[name] = projected_grad * z/batch_size
                else:
                    estimator[name] += projected_grad * z/batch_size
                
        return estimator
    
    def LS_true_grad(self, parameters, x, y):
        grad = {}
        for name, param in parameters:
            grad[name] = (x.T @ x @ param.data.T - x.T @ y)*2/x.shape[0]
        return grad 

    def training_step_ZO_SVRG_Rand_Coord(self, model, batch, epoch, batch_idx):
        # run ZO-SVRG-Rand-Coord update
        x, y = batch
        start = time.time()
        random_seed = np.random.randint(1000000000)
        parameters = []
        for name, param in model.named_parameters():
            parameters.append((name, param))
            
        if epoch % self.q == 0 and batch_idx % 2 == 0:
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
        
        end = time.time()
        self.time.append(end-start)
        self.query.append(2*n)
        
        with torch.no_grad():
            # loss computation
            loss = self.zo_forward(model, x, y)
        print("train_loss", loss)
        self.tr_loss.append(loss.detach().cpu().numpy())
        return loss

    def training_step_ZO_SVRG(self, model, batch, epoch, batch_idx):
        # run ZO-SVRG update
        start = time.time()
        x, y = batch
        parameters = []
        for name, param in model.named_parameters():
            parameters.append((name, param))
            
        if batch_idx % self.q == 0:
            n = self.X.shape[0]
            self.full_grad = self.SPSA_estimator(model, parameters, x, y)
            with torch.no_grad():
                for name, param in parameters:
                    param.data = param.data - self.learning_rate_aux * self.full_grad[name]
            self.parameters = parameters.copy()
        else:
            n = x.size(0)
            f_rand_curr = self.SPSA_estimator(model, parameters, x, y, fullbatch=False)
            f_rand_past = self.SPSA_estimator(model, self.parameters, x, y, fullbatch=False)
            v = {}
            for name, param in parameters:
                v[name] = f_rand_curr[name] - f_rand_past[name] + self.full_grad[name]
                param.data = param.data - self.learning_rate * v[name]
        end = time.time()
        self.time.append(end-start)
        self.query.append(2*n)
        
        with torch.no_grad():
            # loss computation
            loss = self.zo_forward(model, x, y)
        print("train_loss", loss)
        self.tr_loss.append(loss.detach().cpu().numpy())
        return loss



    def training_step_ZO(self, model, batch):
        # run ZO update
        start = time.time()
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
        
        self.abs_proj.append(torch.abs(projected_grad))
        
        # reset model back to its parameters at start of step
        self.efficient_perturb_parameters(parameters, random_seed)
        torch.manual_seed(random_seed)
        #print(parameters)
        for name, param in parameters:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            g_estimate = projected_grad * z
            g_true = (x.T @ x @ param.data.T - x.T @ y)*2/x.shape[0]
            #print(g_estimate.shape)
            #print(g_true.shape)
            dot_product = torch.dot(torch.squeeze(g_estimate), torch.squeeze(g_true)) 
            param.data = param.data - self.learning_rate * g_estimate
        end = time.time()
        self.time.append(end-start)
        self.query.append(2*n)
        
        
        with torch.no_grad():
            # loss computation
            loss = self.zo_forward(model, x, y)
        print("train_loss", loss)
        #print("Grad Cosine Sim", dot_product)
        self.tr_loss.append(loss.detach().cpu().numpy())
        self.grad_dot.append(dot_product.detach().cpu().numpy())
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        n = x.size(0)
        start = time.time()
        # Run FO update
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        self.tr_loss.append(loss.detach().cpu().numpy())
        end = time.time()
        self.time.append(end-start)
        self.query.append(n)
        return loss
                
    def validation_step_ZO(self, model, x, y):        
        loss = nn.functional.mse_loss(model(x), y)
        print("valid_loss", loss)
        return loss              
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
            

