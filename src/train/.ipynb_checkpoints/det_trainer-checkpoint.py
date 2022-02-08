import sys
import pickle
import copy
import warnings
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW

from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from oil.tuning.study import train_trial
from oil.utils.utils import LoaderTo, islice, \
							FixedNumpySeed, cosLr, \
							Eval, export
from oil.model_trainers import Trainer

from ..systems.chain_pendulum import ChainPendulum
from ..systems.rigid_body import project_onto_constraints
from ..models import HNN,NN,CHNN,CH
from ..datasets import RigidBodyDataset

import src.datasets as datasets
import src.models as models
import src.systems as systems

def logspace(a, b, k):
    return np.exp(np.linspace(np.log(a), np.log(b), k))

class IntegratedDynamicsTrainer(Trainer):
    """ Model should specify the dynamics, mapping from t,z -> dz/dt"""

    def __init__(self, *args, tol=1e-4, constrained=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers["tol"] = tol
        self.num_mbs = 0
        self.constrained = constrained
        self.collect = False
        self.prob_loss = False

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        self.num_mbs += 1
        (z0, ts), true_zs = minibatch
        pred_zs = self.model.integrate(z0, ts[0], tol=self.hypers["tol"])
        return (pred_zs - true_zs).abs().mean()

    def nll_loss(self, minibatch):
        self.num_mbs += 1
        (z0, ts), true_zs = minibatch
        nll = self.model.nll(true_zs, z0, ts[0], self.hypers["tol"])

        print("NLL: {}".format(nll))

        return nll

    def step(self, minibatch):
        self.optimizer.zero_grad()
        if self.prob_loss:
            loss = self.nll_loss(minibatch)
        else:
            loss = self.loss(minibatch)
        loss.backward()
        self.optimizer.step()
        
        if self.collect:
            self.model.collect_model()
        
        return loss

    def metrics(self, loader):
        mae = lambda mb: self.loss(mb).cpu().data.numpy()
        return {"MAE": self.evalAverageMetrics(loader, mae)}

    def logStuff(self, step, minibatch=None):
        self.logger.add_scalars(
            "info", {"nfe": self.model.nfe / (max(self.num_mbs, 1e-3))}, step
        )
        super().logStuff(step, minibatch)

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
    lr=3e-3, bs=200, num_epochs=100, trainer_config={}, net_seed=None, n_subsample=None,
    noise_rate=None):
    
    # Create Training set and model
    if isinstance(network, str):
        network = eval(network)
    angular = not issubclass(network,CH)

    if n_subsample is None:
        n_subsample = n_systems
    splits = {"train": int(0.8 * n_subsample), "test": int(0.2 * n_subsample)}
    body.integration_time = tau
    with FixedNumpySeed(0):
        dataset = dataset(root_dir=root_dir, n_systems=n_systems, regen=regen,
                          chunk_len=C, body=body, angular_coords=angular,
                          n_subsample=n_subsample, noise_rate=noise_rate)
        datasets = split_dataset(dataset, splits)

    dof_ndim = dataset.body.D if angular else dataset.body.d
    
    model = network(dataset.body.body_graph, dof_ndim=dof_ndim,
                    angular_dims=dataset.body.angular_dims, **net_cfg)
    if torch.cuda.is_available() and device is None:
      device = "cuda"
    model = model.float().to(device)

    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {
        k: LoaderTo(DataLoader(v, batch_size=min(bs, splits[k]), num_workers=0, 
            shuffle=(k == "train")), device=device, dtype=torch.float32)
        for k, v in datasets.items()}

    dataloaders["Train"] = dataloaders["train"]

    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: AdamW(params, lr=lr, weight_decay=1e-5)
    lr_sched = cosLr(num_epochs)
    return IntegratedDynamicsTrainer(
        model, dataloaders, opt_constr, lr_sched,
        constrained=issubclass(network, CH), log_args={"timeFrac": 1 / 4, "minPeriod": 0.0},
        **trainer_config)
