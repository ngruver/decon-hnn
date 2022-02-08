import os
import re
import sys
import copy
import glob
import wandb
import subprocess
import numpy as np
from functools import partial

import torch
import torch.nn as nn

from .det_trainer import make_trainer as make_det_trainer
from ..uncertainty.swag import SWAG

class SWAGModel(nn.Module):

    def __init__(self, model, **kwargs):
        super().__init__()

        self.model = model(**kwargs)
        self.swag_model = SWAG(model, **kwargs)
        self.nfe = self.model.nfe

    def integrate(self, z, t, tol):
        if self.training:
            return self.model.integrate(z, t, tol=tol)
        else:
            return self(z, t).mean(0)

    def forward(self, z, t, n_samples=10):
        pred_zt = []
        for _ in range(n_samples):
            self.swag_model.sample()
            with torch.no_grad():
                zt_pred = self.swag_model.base.integrate(z, t, method='rk4')
            pred_zt.append(zt_pred)
        pred_zt = torch.stack(pred_zt, dim=0)
        return pred_zt

    def collect_model(self):
        self.swag_model.collect_model(self.model)

class SWAGTrainer():

    def __init__(self, swag_epochs=20, **kwargs):
        kwargs['network'] = partial(SWAGModel, kwargs['network'])
        self._trainer = make_det_trainer(**kwargs)
        self.model = self._trainer.model
        self.swag_epochs = swag_epochs

    def train(self, num_epochs):
        self._trainer.train(num_epochs)
        self._trainer.collect = True
        self._trainer.train(self.swag_epochs)

class DeepEnsembleModel(nn.Module):
    
    def __init__(self, model, ensemble, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ensemble = ensemble

    # def forward(self, z, m, t, n_samples=10, statewise=False):
    def forward(self, z, t, n_samples=10, statewise=False):
        if statewise:
            with torch.no_grad():
                pred_zt = self.model.integrate(z, m, t, method='rk4', models=self.ensemble)
            return pred_zt
        else:
            pred_zt = []             
            for model in self.ensemble:
                with torch.no_grad():
                    zt_pred = model.integrate(z, t, method='rk4')
                    # zt_pred = model.integrate(z, m, t, method='rk4')
                pred_zt.append(zt_pred)
            pred_zt = torch.stack(pred_zt, dim=0)
            return pred_zt

class DeepEnsembleTrainer():

    def __init__(self, ensemble_size=5, num_bodies=2, **kwargs):
        self.ensemble_size = ensemble_size

        self._trainers = [make_det_trainer(**kwargs) for _ in range(self.ensemble_size)]
        self.ensemble = nn.ModuleList([copy.deepcopy(t.model) for t in self._trainers])
        
        _model = make_det_trainer(**kwargs).model
        self.model = DeepEnsembleModel(_model, self.ensemble)

    def train(self, num_epochs):
        self.ensemble = nn.ModuleList([])
        for idx, trainer in enumerate(self._trainers):
            print(f"Training ensemble member {idx}...")
            trainer.train(num_epochs)
            self.ensemble.append(copy.deepcopy(trainer.model))
        self.model.ensemble = self.ensemble

class AleotoricWrapper(nn.Module):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, z, t, n_samples=10):
        self.model.eval()
        with torch.no_grad():
            zt_pred = self.model.integrate(z, t, method='rk4')
            var = self.model.get_covariance(zt_pred, t).reshape(*zt_pred.shape)
        return zt_pred, var

class AleotoricTrainer():

    def __init__(self, **kwargs):
        self._trainer = make_det_trainer(**kwargs)
        self._trainer.prob_loss = True
        self.model = AleotoricWrapper(self._trainer.model)

    def train(self, num_epochs):
        self._trainer.train(num_epochs) 

class DeterministicWrapper(nn.Module):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward(self, z, t, n_samples=10):
        self.model.eval()
        with torch.no_grad():
            zt_pred = self.model.integrate(z, t, method='rk4')
        return zt_pred

class DeterministicTrainer():

    def __init__(self, **kwargs):
        self._trainer = make_det_trainer(**kwargs)
        self.model = DeterministicWrapper(self._trainer.model)

    def train(self, num_epochs):
        self._trainer.train(num_epochs)

def make_trainer(uq_type=None, **kwargs):
    if uq_type == 'swag':
        kwargs.pop('num_bodies', None)
        return SWAGTrainer(**kwargs)
    elif uq_type == 'deep-ensemble':
        return DeepEnsembleTrainer(**kwargs)
    elif uq_type == 'deep-ensemble-step':
        return DeepEnsembleTrainer(**kwargs)
    elif uq_type == 'output-uncertainty':
        kwargs.pop('num_bodies', None)
        return AleotoricTrainer(**kwargs)
    elif uq_type == 'cnf':
        kwargs.pop('num_bodies', None)
        return DeterministicTrainer(**kwargs)
    elif (uq_type == 'det') or (uq_type is None):
        kwargs.pop('num_bodies', None)
        return DeterministicTrainer(**kwargs)
    else:
        raise NotImplementedError