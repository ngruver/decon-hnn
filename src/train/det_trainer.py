import os
import sys
import wandb
import pickle
import copy
import shutil
import warnings
import tempfile
import numpy as np
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
from torch.nn.utils import clip_grad_norm_

from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from oil.tuning.study import train_trial
from oil.utils.utils import LoaderTo, islice, \
							FixedNumpySeed, cosLr, \
							Eval, export
from oil.model_trainers import Trainer

from ..systems.chain_pendulum import ChainPendulum
from ..systems.spring_pendulum import SpringPendulum
from ..systems.rigid_body import project_onto_constraints
from ..models import HNN,NN,CHNN,CH,mNN,FlexHNN
from ..models.rpp_hnn import FlexHNN, ControlFlexHNN, MixtureHNN
from ..datasets import RigidBodyDataset, PlanetaryDataset
from ..uncertainty.swag import SWAG

import src.datasets as datasets
import src.models as models
import src.systems as systems
from src.models.utils import convert_linear_to_snlinear

def logspace(a, b, k):
    return np.exp(np.linspace(np.log(a), np.log(b), k))

class IntegratedDynamicsTrainer(Trainer):
    """ Model should specify the dynamics, mapping from t,z -> dz/dt"""

    def __init__(self, *args, tol=1e-4, constrained=False, loss="l2", **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers["tol"] = tol
        self.num_mbs = 0
        self.constrained = constrained
        self.collect = False
        self.prob_loss = False
        self.loss_type = loss
        self.body = self.dataloaders["train"].dataset.body
        if "ms" in dir(self.body):
            self.m = torch.from_numpy(np.array(self.body.ms))
            print(f"***MASS***: {self.m}")

    def loss(self, minibatch, regularization=True):
        with torch.enable_grad():
            """ Standard cross-entropy loss """
            self.num_mbs += 1
            (z0, ts), true_zs = minibatch
            pred_zs = self.model.integrate(z0, ts[0], tol=self.hypers["tol"])
            if self.loss_type == "l2":
                loss = (pred_zs - true_zs).pow(2).mean()
            elif self.loss_type == "l1":
                loss = (pred_zs - true_zs).abs().mean()

            ts = ts[0]
            q, v = true_zs[:,:,0], true_zs[:,:,1]
            # print(f"body dt: {self.body.dt, self.body.integration_time}")
            # print(f"delta t: {ts[1] - ts[0]}")
            # print(f"positions: {q[0]}")
            # print(f"velocities: {v[0]}")
            # print(f"temporal diffs: {torch.diff(q, dim=1)[0] / (ts[1] - ts[0])}")
            
            # lstsq_fit = np.linalg.lstsq(np.vstack([v[:,:-1].flatten(), \
            #                             np.ones(len(v[:,:-1].flatten()))]).T, \
            #                             torch.diff(q, dim=1).flatten(), rcond=None)[0]
            # print(f"lsrsq: {lstsq_fit}")
            # print("\n")

            # if regularization and "rpp_hnn" in str(type(self.model)):
            #     loss += self.model.regularization(self)
            if hasattr(self.model, "regularization"):
                loss += self.model.regularization(self)
            return loss

    def nll_loss(self, minibatch):
        self.num_mbs += 1
        (z0, ts), true_zs = minibatch
        nll = self.model.nll(true_zs, z0, ts[0], self.hypers["tol"])

        print("NLL: {}".format(nll))

        return nll

    def step(self, minibatch):
        self.model.train()
        self.optimizer.zero_grad()
        if self.prob_loss:
            loss = self.nll_loss(minibatch)
        else:
            loss = self.loss(minibatch)
        loss.backward()

        # for name, p in self.model.named_parameters():
        #     # if not name.endswith("_s"):
        #     #     continue
        #     if p.grad is None:
        #         continue
        #     print(f"{name} grad norm: {p.grad.pow(2).mean()}")#.pow(2).max()}")
        #     print(f"{name} val norm: {p.data.pow(2).mean().item()}")
        # print("")

        clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        
        if self.collect:
            self.model.collect_model()
        
        return loss

    # def energy_conservation(self, pred_zs):
    #     _pred_zs = self.body.body2globalCoords(pred_zs.reshape(-1, *pred_zs.shape[2:])).cpu()
    #     energy = self.body.total_energy(_pred_zs).reshape(*pred_zs.shape[:2])
    #     square_err = (energy - energy[:,:1]).pow(2)
    #     norm_term = energy.pow(2) + energy[:,:1]
    #     return self.loggeomean_rel_err(square_err, norm_term)

    # def loggeomean_rel_err(self, sq_err, norm_term):
    #     rel_err = torch.sqrt(sq_err) / norm_term
    #     loggeomean_rel_err = torch.log(torch.clamp(rel_err,min=1e-7)).mean()
    #     return loggeomean_rel_err

    def rel_err(self, pred_zs, true_zs, ts):
        square_err = ((true_zs - pred_zs)**2).mean((-1,-2))
        rel_err = torch.sqrt(square_err) / \
            (torch.sqrt((true_zs**2).mean((-1,-2)))+torch.sqrt((pred_zs**2).mean((-1,-2))))
        loggeomean_rel_err = torch.log(torch.clamp(rel_err,min=1e-7))
        return loggeomean_rel_err

    def rel_err_vec(self):
        test_loader = self.dataloaders["test"]
        log_H_err = []
        for mb in test_loader:
            self.model.eval()
            (z0, ts), true_zs = mb
            pred_zs = self.model.integrate(z0, ts[0], tol=self.hypers["tol"])
            _log_H_err = self.rel_err(pred_zs, true_zs, ts).cpu().data.numpy()
            log_H_err.append(_log_H_err)
        log_H_err = np.concatenate(log_H_err, axis=0)
        return log_H_err

    def energy_err(self, pred_zs, true_zs, ts):
        bs,T = true_zs.shape[:2]
        cartesian_pred_zs = self.body.body2globalCoords(pred_zs.reshape(bs*T, *pred_zs.shape[2:]))
        pred_H = self.body.hamiltonian(ts, cartesian_pred_zs.reshape(bs*T,-1).cpu())
        cartesian_true_zs = self.body.body2globalCoords(true_zs.reshape(bs*T, *true_zs.shape[2:]))
        true_H = self.body.hamiltonian(ts, cartesian_true_zs.reshape(bs*T,-1).cpu())
        H_err = torch.abs(true_H - pred_H)/(torch.abs(true_H)+torch.abs(pred_H))
        log_H_err = torch.log(torch.clamp(H_err,min=1e-7))
        log_H_err = log_H_err.reshape(bs, T)
        return log_H_err

    def energy_err_vec(self):
        test_loader = self.dataloaders["test"]
        log_H_err = []
        for mb in test_loader:
            self.model.eval()
            (z0, ts), true_zs = mb
            pred_zs = self.model.integrate(z0, ts[0], tol=self.hypers["tol"])
            _log_H_err = self.energy_err(pred_zs, true_zs, ts).cpu().data.numpy()
            log_H_err.append(_log_H_err)
        log_H_err = np.concatenate(log_H_err, axis=0)
        return log_H_err

    def metrics(self, loader):
        def _metrics(mb):
            self.model.eval()
            (z0, ts), true_zs = mb
            pred_zs = self.model.integrate(z0, ts[0], tol=self.hypers["tol"])
            square_err = ((true_zs - pred_zs)**2).mean((-1,-2))
            rel_err = torch.sqrt(square_err) / \
                (torch.sqrt((true_zs**2).mean((-1,-2)))+torch.sqrt((pred_zs**2).mean((-1,-2))))
            loggeomean_rel_err = self.rel_err(pred_zs, true_zs, ts).mean()
            log_H_err = self.energy_err(pred_zs, true_zs, ts).mean()
            return np.array([loggeomean_rel_err.cpu().data.numpy(), square_err.mean().cpu().data.numpy(),
                            log_H_err.cpu().data.numpy()])
        loggeomean,mse,log_herr = self.evalAverageMetrics(loader, _metrics)
        return {"MSE": mse,'gerr':np.exp(loggeomean)}#,'Herr':np.exp(log_herr)}#,

    def logStuff(self, step, minibatch=None):
        self.logger.add_scalars(
            "info", {"nfe": self.model.nfe / (max(self.num_mbs, 1e-3))}, step
        )
        super().logStuff(step, minibatch)
        
        df = self.logger.scalar_frame.iloc[-1:]
        # print(df)
        log_dict = {str(c): df[c].to_numpy()[0] for c in df.columns}

        # err_vec = self.energy_err_vec()[:50]
        # log_dict["H_err_vec"] = wandb.Table(columns=list(range(err_vec.shape[1])), 
        #                                     data=err_vec)

        # err_vec = self.rel_err_vec()[:50]
        # log_dict["rel_err_vec"] = wandb.Table(columns=list(range(err_vec.shape[1])), 
        #                                       data=err_vec)

        try:
            wandb.log(log_dict)
        except Exception as e:
            print(e)
            pass

    def test_rollouts(self, angular_to_euclidean=False, pert_eps=1e-4):
        #self.model.cpu().double()
        dataloader = self.dataloaders["test"]
        rel_errs = []
        pert_rel_errs = []
        with Eval(self.model), torch.no_grad():
            for mb in dataloader:
                z0, T = mb[0]  # assume timesteps evenly spaced for now
                #z0 = z0.cpu().double()
                T = T[0]
                body = dataloader.dataset.body
                long_T = torch.arange(0., 10., body.dt).to(z0.device, z0.dtype)
                
                zt_pred = self.model.integrate(z0, long_T, method='rk4')#long_T, method='rk4')
                bs, Nlong, *rest = zt_pred.shape
                # add conversion from angular to euclidean
                
                if angular_to_euclidean:
                    z0 = body.body2globalCoords(z0)
                    flat_pred = body.body2globalCoords(zt_pred.reshape(bs * Nlong, *rest))
                    zt_pred = flat_pred.reshape(bs, Nlong, *flat_pred.shape[1:])
                
                zt = dataloader.dataset.body.integrate(z0, long_T)
                # perturbation = pert_eps * torch.randn_like(z0) # perturbation does not respect constraints
                # z0_perturbed = project_onto_constraints(body.body_graph,z0 + perturbation,tol=1e-5) #project
                # zt_pert = body.integrate(z0_perturbed, long_T)
                # (bs,T,2,n,2)
                rel_error = ((zt_pred - zt) ** 2).sum(-1).sum(-1).sum(-1).sqrt() / (
                    (zt_pred + zt) ** 2
                ).sum(-1).sum(-1).sum(-1).sqrt()
                rel_errs.append(rel_error)
                # pert_rel_error = ((zt_pert - zt) ** 2).sum(-1).sum(-1).sum(-1 \
                # ).sqrt() / ((zt_pert + zt) ** 2).sum(-1).sum(-1).sum(-1).sqrt()
                # pert_rel_errs.append(pert_rel_error)
            rel_errs = torch.cat(rel_errs, dim=0)  # (D,T)
            # pert_rel_errs = torch.cat(pert_rel_errs, dim=0)  # (D,T)
            both = (rel_errs, zt_pred)#, pert_rel_errs, zt_pred, zt_pert)
        return both

def make_trainer(*,
    network=HNN, net_cfg={}, device=None, root_dir=None,
    dataset=RigidBodyDataset, body=ChainPendulum(3), tau=3, n_systems=1000, regen=False, C=5,
    lr=3e-3, bs=200, num_epochs=100, trainer_config={}, net_seed=0, n_subsample=None, data_seed=0,
    noise_rate=None, weight_decay=1e-4):
    
    # Create Training set and model
    if isinstance(network, str):
        network = eval(network)
    
    if isinstance(network, partial):
        constrained = issubclass(network.args[0],CH)
    else:
        constrained = issubclass(network,CH)
    angular = not constrained

    if n_subsample is None:
        n_subsample = n_systems
    splits = {"train": int(0.8 * n_subsample), "test": int(0.2 * n_subsample)}
    
    body.integration_time = tau
    with FixedNumpySeed(data_seed):
        dataset_cons = dataset
        dataset = dataset_cons(root_dir=root_dir, n_systems=n_systems, regen=regen,
                               chunk_len=C, body=body, angular_coords=angular,
                               n_subsample=n_subsample, noise_rate=noise_rate, seed=data_seed)

        datasets = split_dataset(dataset, splits)

        if isinstance(dataset,PlanetaryDataset):
            datasets["test"] = dataset_cons(root_dir=root_dir, n_systems=splits["test"], regen=regen,
                                            chunk_len=100, body=body, angular_coords=angular,
                                            n_subsample=splits["test"], noise_rate=noise_rate, mode="val",
                                            seed=data_seed, 
                                            z_mean=datasets["train"].z_mean, z_std=datasets["train"].z_std,
                                            m_mean=datasets["train"].m_mean, m_std=datasets["train"].m_std)
        else:
            datasets["test"] = dataset_cons(root_dir=root_dir, n_systems=splits["test"], regen=regen,
                                            chunk_len=100, body=body, angular_coords=angular,
                                            n_subsample=splits["test"], noise_rate=noise_rate, mode="val",
                                            seed=data_seed)

    dof_ndim = dataset.body.D if angular else dataset.body.d
    
    torch.manual_seed(net_seed)
    model = network(G=dataset.body.body_graph, dof_ndim=dof_ndim,
                    angular_dims=dataset.body.angular_dims, **net_cfg)

    if torch.cuda.is_available() and device is None:
      device = "cuda"
    model = model.float().to(device)

    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {
        k: LoaderTo(DataLoader(v, batch_size=min(bs, splits[k]), num_workers=0, 
            shuffle=(k == "train"), drop_last=True), device=device, dtype=torch.float32)
        for k, v in datasets.items()}

    dataloaders["Train"] = dataloaders["train"]

    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: AdamW(params, lr=lr, weight_decay=weight_decay)
    lr_sched = cosLr(num_epochs) #1e8) #cosLr(num_epochs)
    
    return IntegratedDynamicsTrainer(
        model, dataloaders, opt_constr, lr_sched,
        log_dir=os.path.join("runs", tempfile.mkdtemp()),
        constrained=constrained, log_args={"timeFrac": 1 / 4, "minPeriod": 0.0},
        **trainer_config)
