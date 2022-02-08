import os
import wandb
import torch
import functools
import numpy as np
import pandas as pd
from fire import Fire
import matplotlib.pyplot as plt
import seaborn as sns

from src.train.ensemble_trainer import make_trainer
from src.models.nn import NN, ControlNN
from src.models.rpp_hnn import FlexHNN, ControlFlexHNN, MixtureHNN
from src.models.sp_regularized_nn import SpNN, MechanicsNN, SecondOrderNN
from src.models.hnn import HNN, NonseparableHNN
from src.systems.magnet_pendulum import MagnetPendulum
from src.systems.chain_pendulum import ChainPendulum
from src.systems.spring_pendulum import SpringPendulum
from src.systems.friction_pendulum import FrictionChainPendulum, FrictionSpringPendulum
from src.systems.control_pendulum import ControlChainPendulum, AlternatingPolicy
from src.systems.rotor import Rotor
from src.systems.gyroscope import Gyroscope
from src.datasets import get_chaotic_eval_dataset, FixedSeedAll, pendulum_near_separatrix
from IPython.display import display, HTML
from src.systems.rigid_body import project_onto_constraints
from src.datasets import get_chaotic_eval_dataset

import warnings
warnings.filterwarnings('ignore')

# system_dict = {
# 	"ChainPendulum": ChainPendulum,
# 	"FrictionChainPendulum": FrictionChainPendulum,
# 	"ControlChainPendulum": ControlChainPendulum
# }

# model_dict = {
# 	"NN": NN,
# 	"ControlNN": ControlNN,
# 	"HNN": HNN,
# 	"NonseparableHNN": NonseparableHNN
# 	"FlexHNN": FlexHNN,
# 	"ControlFlexHNN": ControlFlexHNN,
# 	"MixtureHNN": MixtureHNN
# }

def main(**cfg):
	print("CUDA AVAILABLE: {}".format(torch.cuda.is_available()))

	run = wandb.init(config=cfg, reinit=True)

	cfg.setdefault("data_seed", 0)
	cfg.setdefault("net_seed", 0)

	cfg.setdefault('device', 'cuda:0' if torch.cuda.is_available() else None)

	# IF CONTROL SETTING, USING LONG TRAJS AND MORE OF THEM
	cfg.setdefault("tau", 10.0)
	cfg.setdefault("C", 5)

	cfg.setdefault("loss", "l2")
	cfg.setdefault("lr", 5e-3)
	cfg.setdefault("bs", 200)
	cfg.setdefault("num_epochs", 2000)
	cfg.setdefault("n_systems", 1000)
	cfg.setdefault("weight_decay", 1e-4)

	cfg.setdefault("uq_type", None)
	cfg.setdefault("regen", False)
	
	system_type = cfg.pop("system_type", "ChainPendulum")
	num_bodies = cfg.pop("num_bodies", 2)
	f_const = cfg.pop("f_const", 0.4)

	system_args = {}
	if system_type.startswith("Friction"):
		system_args["f_const"] = f_const

	body = eval(system_type)(num_bodies, **system_args)
	
	net_cfg = {}
	model_type = cfg.pop("model_type", "NN")
	if model_type == "FlexHNN" or model_type == "MixtureHNN":
		alpha_beta = cfg.pop("alpha_beta", None)
		if alpha_beta:
			alpha, beta = alpha_beta
			net_cfg["alpha"] = alpha, 
			net_cfg["beta"] = beta
		else:
			net_cfg["alpha"] = cfg.pop("alpha", 1e6) 
			net_cfg["beta"] = cfg.pop("beta", 1e6)
		cfg.setdefault("net_cfg", net_cfg)

	if model_type == "MixtureHNN":
		tie_layers = cfg.pop("tie_layers", 0)
		net_cfg["tie_layers"] = tie_layers
		cfg.setdefault("net_cfg", net_cfg)

	if system_type == "ControlChainPendulum":
		if model_type == "FlexHNN":
			model_type = "ControlFlexHNN"
		elif model_type == "NN":
			model_type = "ControlNN"

		if cfg["C"] < 10:
			cfg["C"] = 10

		if cfg["n_systems"] < 2000:
			cfg["n_systems"] = 2000

	print(model_type)
	net = eval(model_type)
	if model_type.startswith("Control"):
		policy = AlternatingPolicy()
		net_cfg["control_policy"] = policy

	net_cfg["hidden_size"] = cfg.pop("hidden_size", 128)

	if "alpha" in cfg:
		net_cfg["alpha"] = cfg.pop("alpha")

	# if model_type == "HNN" and num_bodies > 2:
	# 	net_cfg["hidden_size"] = 64
	# 	cfg["weight_decay"] = 1e-1

	cfg["net_cfg"] = net_cfg

	if model_type == "HNN" and system_type == "ControlChainPendulum":
		cfg["lr"] = 1e-4

	from pprint import pprint
	pprint(cfg)

	trainer = make_trainer(**cfg, network=net, body=body, 
		trainer_config=dict(loss=cfg.pop("loss", "l2")))

	# if model_type == "MixtureHNN":
	# 	convert_linear_to_snlinear(trainer.model)
	# 	trainer.model = trainer.model.cuda()

	trainer.train(cfg["num_epochs"])

	run.finish()

if __name__ == "__main__":
	# os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

	Fire(main)