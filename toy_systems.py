import wandb
import torch
from pprint import pprint
from fire import Fire

from src.train.trainer import make_trainer
from src.models import NN, HNN, NonseparableHNN, MixtureHNN, \
					   SpNN, MechanicsNN, SecondOrderNN
from src.systems import MagnetPendulum, ChainPendulum, SpringPendulum, \
						FrictionChainPendulum, FrictionSpringPendulum, \
						Rotor, Gyroscope

import warnings
warnings.filterwarnings('ignore')

def main(**cfg):
	print("CUDA AVAILABLE: {}".format(torch.cuda.is_available()))

	run = wandb.init(config=cfg, reinit=True)

	cfg.setdefault("data_seed", 0)
	cfg.setdefault("net_seed", 0)

	cfg.setdefault('device', 'cuda:0' if torch.cuda.is_available() else None)

	cfg.setdefault("tau", 10.0)
	cfg.setdefault("C", 5)

	cfg.setdefault("loss", "l2")
	cfg.setdefault("lr", 5e-3)
	cfg.setdefault("bs", 200)
	cfg.setdefault("num_epochs", 2000)
	cfg.setdefault("n_systems", 1000)
	cfg.setdefault("weight_decay", 1e-4)

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

	print(model_type)
	net = eval(model_type)

	net_cfg["hidden_size"] = cfg.pop("hidden_size", 128)
	cfg["net_cfg"] = net_cfg

	pprint(cfg)

	cfg.pop('num_bodies', None)
	trainer = make_trainer(**cfg, network=net, body=body, 
		trainer_config=dict(loss=cfg.pop("loss", "l2")))

	trainer.train(cfg["num_epochs"])

	run.finish()

if __name__ == "__main__":
	Fire(main)