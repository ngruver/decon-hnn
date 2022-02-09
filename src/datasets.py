import os
import sys
import tqdm
import random
import numpy as np
import networkx as nx
import glob
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset

from scipy.optimize import minimize_scalar

from .systems.rigid_body import RigidBody
from .systems.rigid_body import project_onto_constraints
from .systems.chain_pendulum import ChainPendulum

class FixedPytorchSeed(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.pt_rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
          torch.cuda.manual_seed(self.seed)
    def __exit__(self, *args):
        torch.random.set_rng_state(self.pt_rng_state)

class FixedSeedAll(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)
        self.pt_rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)
        torch.random.set_rng_state(self.pt_rng_state)

def rel_err(x: Tensor, y: Tensor) -> Tensor:
    return (((x - y) ** 2).sum() / ((x + y) ** 2).sum()).sqrt()

class RigidBodyDataset(Dataset):
    space_dim = 2
    num_targets = 1

    def __init__(
        self,
        root_dir=None,
        body=ChainPendulum(3),
        n_systems=100,
        regen=False,
        chunk_len=5,
        angular_coords=False,
        seed=0,
        mode="train",
        n_subsample=None,
        noise_rate=None,
        z_0_mask_idxs=None,
        z_mask_idxs=None,
        **kwargs
    ):
        super().__init__()
        with FixedSeedAll(seed):
            self.mode = mode
            root_dir = root_dir or os.path.join(os.environ['DATADIR'], 'ODEDynamics', self.__class__.__name__)
            self.body = body
            tag = f"{body.__class__.__name__}_{body.n}_N{n_systems}_{mode}_{seed}"
            if body.__class__.__name__.startswith("Friction"):
                tag = f"{body.__class__.__name__}_{body.n}_{body.f_const}_N{n_systems}_{mode}_{seed}"
            filename = os.path.join(
                root_dir, f"trajectories_{tag}.pz"
            )
            if os.path.exists(filename) and not regen:
                ts, zs = torch.load(filename)
            else:
                ts, zs = self.generate_trajectory_data(n_systems)
                os.makedirs(root_dir, exist_ok=True)
                torch.save((ts, zs), filename)

            Ts, Zs = self.chunk_training_data(ts, zs, chunk_len)

            if n_subsample is not None:
                Ts, Zs = Ts[:n_subsample], Zs[:n_subsample]
    
            self.Ts, self.Zs = Ts.float(), Zs.float()

            self.seed = seed
    
            if angular_coords:
                N, T = self.Zs.shape[:2]
                flat_Zs = self.Zs.reshape(N * T, *self.Zs.shape[2:])
                self.Zs = self.body.global2bodyCoords(flat_Zs.double())
                print(rel_err(self.body.body2globalCoords(self.Zs), flat_Zs))
                self.Zs = self.Zs.reshape(N, T, *self.Zs.shape[1:]).float()


            if isinstance(noise_rate, float):
                self.Zs = self.Zs + noise_rate * torch.randn_like(self.Zs)
   
            self.z_0_mask_idxs = z_0_mask_idxs
            self.z_mask_idxs = z_mask_idxs

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        z_0 = self.Zs[i, 0]
        z = self.Zs[i]
        if self.z_0_mask_idxs is not None:
            z_0[1, self.z_0_mask_idxs] = 0.
            z[0, 1, self.z_0_mask_idxs] = 0.
        if self.z_mask_idxs is not None:
            z[1:, 1, self.z_mask_idxs] = 0.
        return (z_0, self.Ts[i]), z

    def generate_trajectory_data(self, n_systems, bs=10000):
        """ Returns ts: (n_systems, traj_len) zs: (n_systems, traj_len, z_dim) """

        def base_10_to_base(n, b):
            """Writes n (originally in base 10) in base `b` but reversed"""
            if n == 0:
                return '0'
            nums = []
            while n:
                n, r = divmod(n, b)
                nums.append(r)
            return list(nums)

        batch_sizes = base_10_to_base(n_systems, bs)
        n_gen = 0
        t_batches, z_batches = [], []
        for i, batch_size in enumerate(batch_sizes):
            if batch_size == 0:
                continue
            batch_size = batch_size * (bs**i)
            print(f"Generating {batch_size} more chunks")
            z0s = self.sample_system(batch_size)
            ts = torch.arange(
                0, self.body.integration_time, self.body.dt, device=z0s.device, dtype=z0s.dtype
            )
            new_zs = self.body.integrate(z0s, ts)
            t_batches.append(ts[None].repeat(batch_size, 1))
            z_batches.append(new_zs)
            n_gen += batch_size
            print(f"{n_gen} total trajectories generated for {self.mode}")
        ts = torch.cat(t_batches, dim=0)[:n_systems]
        zs = torch.cat(z_batches, dim=0)[:n_systems]
        return ts, zs

    def chunk_training_data(self, ts, zs, chunk_len):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (batch_size, traj_len)] [zs (batch_size, traj_len, *z_dim)]
        outputs: [chosen_ts (batch_size, chunk_len)] [chosen_zs (batch_size, chunk_len, *z_dim)]"""
        n_trajs, traj_len, *z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        # Cut each trajectory into non-overlapping chunks
        chunks_ts = ts.split(chunk_len, dim=1)
        chunks_zs = zs.split(chunk_len, dim=1)

        if (traj_len % chunk_len) != 0:
            chunks_ts = chunks_ts[:-1]
            chunks_zs = chunks_zs[:-1]
            n_chunks = n_chunks - 1

        chunked_ts = torch.stack(chunks_ts)
        chunked_zs = torch.stack(chunks_zs)

        # From each trajectory, we choose a single chunk randomly
        chunk_idx = torch.randint(0, n_chunks, (n_trajs,), device=zs.device).long()
        chosen_ts = chunked_ts[chunk_idx, range(n_trajs)]
        chosen_zs = chunked_zs[chunk_idx, range(n_trajs)]

        return chosen_ts, chosen_zs

    def sample_system(self, N):
        """"""
        return self.body.sample_initial_conditions(N)
