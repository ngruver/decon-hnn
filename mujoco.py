import wandb
import pprint
import numpy as np
import pandas as pd
from fire import Fire
import random

import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.nn.utils import clip_grad_norm_

from src.models.dynamics_models import RecurrentNetwork, \
									   NODE, CoupledNODE, \
									   RecNODE, CoupledRecNODE, \
									   RPPNet, MechanicsNN, \
									   MixtureHNN

timestep_table = {
	'Humanoid-v2': 0.015,
	'Walker2dFull-v0': .008,
	'Ant-v2': .05,
	'SwimmerFull-v0': .04,
	'Hopper-v2': .008,
	'HalfCheetahFull-v0': .05,
	'HopperFull-v0': 0.008,
}


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_net(net, data, z_mean, z_std, num_epochs, weight_decay=1e-6, log_freq=1):
	(train_loader, test_loader, train_ts, test_ts) = data

	optimizer = optim.Adam(net.param_groups, lr=5e-3, weight_decay=weight_decay)
	lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

	records = []
	for epoch_idx in range(num_epochs):

		net.train()
		avg_loss = 0
		avg_grad_norm = 0
		for x, u in train_loader:

			if torch.cuda.is_available():
				x, u = x.to('cuda:0'), u.to('cuda:0')

			z_pred = net.integrate(x[:,0], train_ts, u)[:, :-1]

			optimizer.zero_grad()
			loss = (z_pred - x).div(z_std).pow(2).mean()
			loss.backward()
			clip_grad_norm_(net.parameters(), 1)
			optimizer.step()

			for name, p in net.named_parameters():
				if p.grad is None:
					continue
				avg_grad_norm += p.data.pow(2).log().sum().item()

			avg_loss += loss.item() / len(train_loader)
			avg_grad_norm /= count_parameters(net) * len(train_loader)

		net.eval()
		test_mse, test_med_se = 0., 0.
		for x, u in test_loader:

			if torch.cuda.is_available():
				x, u = x.to('cuda:0'), u.to('cuda:0')

			with torch.no_grad():
				z_pred = net.integrate(x[:,0], test_ts, u)[:, :-1]

			test_se = (z_pred - x).div(z_std).pow(2).mean(0).mean(-1)
			test_mse += test_se.mean().item() / len(test_loader)
			test_med_se += test_se.median().item() / len(test_loader)

		metric_dict = dict(train_mse=avg_loss, test_mse=test_mse, test_med_se=test_med_se,
						   epoch=epoch_idx)

		# print(f"grad norm: {np.exp(avg_grad_norm)}\n")

		lr_sched.step()
		records.append(metric_dict)

		if (epoch_idx + 1) % log_freq == 0:
			try:
				pprint.pprint(metric_dict)
				wandb.log(metric_dict)
			except Exception:
				pass

	return records


def seq_rmse(net, loader, ts, z_std):
	rmse = 0.
	for z, u in loader:
		z, u = z.to('cuda:0'), u.to('cuda:0')

		with torch.no_grad():
			z_pred = net.integrate(z[:,0], ts, u)[:, :-1]

		test_se = (z_pred - z).div(z_std).pow(2).mean(0).mean(-1)
		rmse += test_se.sqrt() / len(loader)
	return rmse


def train_mujoco_model(cfg):
	random.seed(cfg['seed'])
	np.random.seed(cfg['seed'])
	torch.manual_seed(cfg['seed'])
	task = cfg["task"]

	train_x = np.load(f"data/{task}_cl20_xdata.npy")
	train_u = np.load(f"data/{task}_cl20_udata.npy")

	test_x = np.load(f"data/{task}_episodes_xdata.npy")
	test_u = np.load(f"data/{task}_episodes_udata.npy")

	subsample_ratio = cfg["subsample_ratio"]
	train_seq_len = cfg["train_seq_len"]

	num_train = int(train_x.shape[0] * subsample_ratio)
	select_idxs = np.random.permutation(train_x.shape[0])[:num_train]
	train_x = train_x[select_idxs][:, :train_seq_len]
	train_u = train_u[select_idxs][:, :train_seq_len]

	num_train, _, x_size = train_x.shape
	_, _, u_size = train_u.shape

	num_test, test_seq_len, _ = test_x.shape

	env_dt = timestep_table[task]

	train_ts = torch.tensor(
	    np.linspace(0, env_dt*(train_seq_len), train_seq_len + 1)
	).float().to('cuda:0')
	test_ts = torch.tensor(
	    np.linspace(0, env_dt*(test_seq_len), test_seq_len + 1)
	).float().to('cuda:0')

	z_mean = torch.tensor(train_x.reshape(-1, x_size).mean(0)).to('cuda:0')
	z_std = torch.tensor(
	    np.clip(train_x.reshape(-1, x_size).std(0), a_min=1e-6, a_max=None)
	).to('cuda:0')

	q_delta = (train_x[:, 1:, :(x_size // 2)] - train_x[:, :-1, :(x_size // 2)])
	q_dot_fd = q_delta / env_dt
	q_dot = (train_x[:, 1:, (x_size // 2):] + train_x[:, :-1, (x_size // 2):]) / 2

	v_delta = (train_x[:, 1:, (x_size // 2):] - train_x[:, :-1, (x_size // 2):])
	v_dot_fd = v_delta / env_dt

	fd_err = np.power(q_dot - q_dot_fd, 2).mean((0,1))
	print(f'dq/dt finite-diff error: {fd_err}') #:0.6f}')  # check that finite difference velocity and observed velocity are close

	lstsq_fit = np.linalg.lstsq(np.vstack([q_dot.flatten(), \
								np.ones(len(q_dot.flatten()))]).T, \
								q_dot_fd.flatten(), rcond=None)[0]
	print(f"lsrsq: {lstsq_fit}")

	train_dataset = data_utils.TensorDataset(
	    torch.tensor(train_x).float(),
	    torch.tensor(train_u).float()
	)

	test_dataset = data_utils.TensorDataset(
	    torch.tensor(test_x).float(),
	    torch.tensor(test_u).float()
	)

	train_loader = data_utils.DataLoader(train_dataset, cfg["bs"])
	test_loader = data_utils.DataLoader(test_dataset, cfg["bs"])

	model_type = cfg["model_type"]
	hidden_size = cfg["hidden_size"]
	weight_decay = cfg["weight_decay"]
	mlp_depth = cfg["mlp_depth"]
	rec_depth = cfg["rec_depth"]

	if model_type == 'RecurrentNetwork':
		model_cfg = dict(input_size=x_size + u_size, output_size=x_size,
						 enc_hidden_size=hidden_size, rec_hidden_size=hidden_size, 
						 dec_hidden_size=hidden_size, enc_depth=mlp_depth // 2, rec_depth=rec_depth,
                         dec_depth=mlp_depth // 2, weight_decay=weight_decay)
	elif model_type == "NODE":
		model_cfg = dict(input_size=x_size + u_size, output_size=x_size,
						 hidden_size=hidden_size, num_layers=mlp_depth, weight_decay=weight_decay,
						 method=cfg['int_method'], step_size=env_dt / cfg['num_int_steps'])
	elif model_type == "CoupledNODE":
		model_cfg = dict(input_size=x_size + u_size, output_size=x_size // 2,
						 hidden_size=hidden_size, num_layers=mlp_depth, weight_decay=weight_decay,
						 method=cfg['int_method'], step_size=env_dt / cfg['num_int_steps'])
	elif model_type == "RecNODE":
		model_cfg = dict(input_size=x_size + u_size, output_size=x_size,
						 enc_hidden_size=hidden_size, rec_hidden_size=hidden_size, 
						 dec_hidden_size=hidden_size, enc_depth=mlp_depth // 2, rec_depth=rec_depth,
                         dec_depth=mlp_depth // 2, weight_decay=weight_decay,
						 method=cfg['int_method'], step_size=env_dt / cfg['num_int_steps'])
	elif model_type == "CoupledRecNODE":
		model_cfg = dict(input_size=x_size + u_size, output_size=x_size // 2,
						 enc_hidden_size=hidden_size, rec_hidden_size=hidden_size, 
						 dec_hidden_size=hidden_size, enc_depth=mlp_depth // 2, rec_depth=rec_depth,
                         dec_depth=mlp_depth // 2, weight_decay=weight_decay,
						 method=cfg['int_method'], step_size=env_dt / cfg['num_int_steps'])
	elif model_type == "MechanicsNN":
		model_cfg = dict(input_size=x_size + u_size, output_size=x_size, 
						 hidden_size=hidden_size, num_layers=mlp_depth, weight_decay=weight_decay,
						 method='euler', step_size=env_dt)
	elif model_type == "MixtureHNN":
		model_cfg = dict(input_size=x_size + u_size, output_size=x_size, 
						 hidden_size=hidden_size, num_layers=mlp_depth, weight_decay=weight_decay,
						 method='euler', step_size=env_dt)	
	else:
		raise ValueError('unrecognized model type')

	net = eval(model_type)(**model_cfg).to('cuda')

	net.input_loc = torch.tensor(
    	np.concatenate((train_x, train_u), axis=-1).reshape(-1, x_size + u_size).mean(0)
	).to(net.input_loc)
	net.input_scale = torch.tensor(
	    np.concatenate((train_x, train_u), axis=-1).reshape(-1, x_size + u_size).std(0)
	).to(net.input_scale)

	if model_type == "RecurrentNetwork":
		net.output_loc = torch.tensor(
		    train_x.reshape(-1, x_size).mean(0)
		).to(net.output_loc)
		net.output_scale = torch.tensor(
		    train_x.reshape(-1, x_size).std(0)
		).to(net.output_scale)
	elif model_type == "NODE" or model_type == "RecNODE":
		net.output_loc = torch.tensor(
		    np.concatenate((q_dot_fd, v_dot_fd), axis=-1).reshape(-1, x_size).mean(0)
		).to(net.output_loc)
		net.output_scale = torch.tensor(
		    np.concatenate((q_dot_fd, v_dot_fd), axis=-1).reshape(-1, x_size).std(0)
		).to(net.output_scale)
	elif model_type == "CoupledRecNODE" or model_type == "CoupledNODE":
		net.output_loc = torch.tensor(
		    v_dot_fd.reshape(-1, x_size // 2).mean(0)
		).to(net.output_loc)
		net.output_scale = torch.tensor(
		    v_dot_fd.reshape(-1, x_size // 2).std(0)
		).to(net.output_scale)
	else:
		pass


	num_epochs = cfg["num_epochs"]
	data = (train_loader, test_loader, train_ts, test_ts)
	records = fit_net(net, data, z_mean, z_std, num_epochs, weight_decay=weight_decay,
					  log_freq=cfg["log_freq"])
	df = pd.DataFrame(records)

	print(df)

def main(**cfg):
	print("CUDA AVAILABLE: {}".format(torch.cuda.is_available()))

	# wandb.init(project='physics-uncertainty-exps', config=cfg)

	cfg.setdefault("exp_group", 'mujoco_dx')
	cfg.setdefault("exp_version", '0.0.4')
	cfg.setdefault("seed", 0)
	cfg.setdefault("log_freq", 8)

	cfg.setdefault("model_type", "NODE")
	cfg.setdefault("task", "HopperFull-v0")
	cfg.setdefault("subsample_ratio", 1.0)
	cfg.setdefault("train_seq_len", 3)

	cfg.setdefault("num_epochs", 256)
	cfg.setdefault("hidden_size", 128)
	cfg.setdefault("mlp_depth", 2)
	cfg.setdefault("rec_depth", 1)
	cfg.setdefault("weight_decay", 1e-4)
	cfg.setdefault('int_method', 'euler')
	cfg.setdefault('num_int_steps', 8)

	cfg.setdefault("lr", 2e-4)
	cfg.setdefault("bs", 200)

	cfg.setdefault("uq_type", None)
	cfg.setdefault("regen", False)
	
	train_mujoco_model(cfg)


if __name__ == "__main__":
	# os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

	Fire(main)