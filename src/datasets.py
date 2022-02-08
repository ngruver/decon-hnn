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
            # if (noise_rate is not None) and noise_rate > 0:
            #     z_var = torch.var(self.Zs)
            #     noise_var = noise_rate * z_var 
            #     noise = noise_rate * z_var * torch.randn_like(self.Zs)
            #     self.Zs += noise
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


class PlanetBody:
    d = 2
    D = 3#9
    angular_dims = []
    body_graph = 3 * [0]  # not needed for now.

class PlanetaryDataset(Dataset):
    def __init__(self, mode='train', root_dir=None, n_systems=0,
                 chunk_len=5, seed=0, n_subsample=None, noise_rate=None,
                 regen=False, body=None, angular_coords=None, 
                 z_mean=None, z_std=None, m_mean=None, m_std=None):
        super().__init__()
        
        self.root_dir = root_dir or os.path.join(os.environ['DATADIR'], 'planetary_integrations')
        # TODO: may want to limit this later?
        self.files = glob.glob(os.path.join(self.root_dir, mode, '*.hdf'))
        if n_systems > 0:
            n_systems = min(n_systems, len(self.files))
        else:
            n_systems = len(self.files)

        self.mode = mode
        self.n_systems = n_systems
        self.body = body or PlanetBody()

        mode_tag = mode.replace("/","_")
        filename = os.path.join(
            self.root_dir, f"trajectories_planetary_N{n_systems}_{chunk_len}_{mode_tag}.pz"
        )
        if os.path.exists(filename) and not regen:
            ts, zs, ms = torch.load(filename)
        else:
            with FixedSeedAll(seed):
                ts, zs, ms = self.create_dataset(N=n_systems, T=chunk_len)
            os.makedirs(self.root_dir, exist_ok=True)
            torch.save((ts, zs, ms), filename)
        
        self.Ts = torch.from_numpy(ts).float()
        self.Zs = torch.from_numpy(zs).float()
        self.Ms = torch.from_numpy(ms).float()

        print(z_mean, z_std)
        if z_mean is None or z_std is None:
            self.z_mean, self.z_std = self.Zs.mean((0,1)), self.Zs.std((0,1))
            # self.z_std = 1 if self.z_std == 0 else self.z_std
            self.z_std[self.z_std == 0] = 1.
        else:
            self.z_mean, self.z_std = z_mean, z_std

        print(f"z_mean: {self.z_mean}, z_std: {self.z_std}")
        self.Zs = (self.Zs - self.z_mean) / self.z_std

        # if m_min is None or m_max is None:
        #     self.m_min, self.m_max = self.Ms.min(0)[0], self.Ms.max(0)[0]
        # else:
        #     self.m_min, self.m_max = m_min, m_max

        # self.Ms = ((self.Ms - self.m_min) / (self.m_max - self.m_min)) + 1e-1

        if m_mean is None or m_std is None:
            self.m_mean, self.m_std = self.Ms.mean(), self.Ms.std() #self.Zs.mean((0,1)), self.Zs.std((0,1))
            self.m_std = 1 if self.m_std == 0 else self.m_std
            # self.z_std[self.z_std == 0] = 1.
        else:
            self.m_mean, self.m_std = m_mean, m_std

        print(f"m_mean: {self.m_mean}, m_std: {self.m_std}")
        self.Ms = (self.Ms - self.m_mean) / self.m_std


        if n_subsample:
            self.Ts, self.Zs, self.Ms = self.Ts[:n_subsample], self.Zs[:n_subsample], self.Ms[:n_subsample]
        if isinstance(noise_rate, float):
            self.Zs = self.Zs + noise_rate * torch.randn_like(self.Zs)

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):        
        # print(self.Zs[i, 0].size())
        # print(self.Ms[i].size())
        # print(self.Ms[i])
        return (self.Zs[i, 0, :, :3], self.Ms[i, :3], self.Ts[i]), self.Zs[i, :, :, :3]

    def inverse_transform(self, z, m):
        return z * self.z_std[None] + self.z_mean[None], \
               (m - 1e-1) * (self.m_max - self.m_min) + self.m_min
               # m * self.m_std[None] + self.m_mean[None]

    @staticmethod
    def load_hdf(fname, mode='cartesian'):
        assert mode in ['orbital', 'cartesian', 'raw']

        masses = ['_'.join([var, str(p)]) for p in range(3) for var in 'm m m'.split(' ')]

        # print(fname)
        traj = pd.read_hdf(fname)

        # print(traj.columns)
        # print(traj["e_0"].values)
        # sys.exit(0)

        if mode == 'raw':
            return traj
        elif mode == 'orbital':
            return traj[['t'] + ['_'.join([var, str(p)]) for p in range(3) for var in 'a e inc Omega pomega theta'.split(' ')]]
        elif mode == 'cartesian':
            return traj[['t'] + ['_'.join([var, str(p)]) for p in range(3) for var in 'x y z vx vy vz'.split(' ')]], traj[masses]
        else:
            raise NotImplementedError

    def create_dataset(self, N=1000, T=100, mode='cartesian'):
        '''
        Samples chunks of length T (i.e (T * dt) units of time) from using the following process:
        1. Sample a trajectory.
        2. Sample a chunk.

        Can be optimized, but is only needed to be run once, runs in reasonable time.

        It may not return exactly N as some trajectories may be shorter than desired T.

        Returns:
            ts: N x T
            zs: N x T x 2 x D
        '''

        period = 30
        t = np.arange(T)[None]
        A = np.random.rand(N, 1)
        offset = 2 * np.pi * np.random.rand(N, 1)
        pos = np.stack([A * np.cos((2 * np.pi * t / period) + offset), 
                        A * np.sin((2 * np.pi * t / period) + offset), np.zeros((N,T))], axis=-1)
        vel = np.stack([A * -np.sin((2 * np.pi * t / period) + offset) * A * (2 * np.pi / period), 
                        A * np.cos((2 * np.pi * t / period) + offset) * A * (2 * np.pi / period), np.zeros((N,T))], axis=-1)
        zs = np.stack([pos, vel], axis=-2)
        ms = np.ones((N, 3))
        ts = np.repeat(t, N, axis=0)

        return ts, zs, ms


        idxs = np.arange(len(self.files))
        num_idxs = len(idxs) + 1

        ts = []
        zs = []
        ms = []
        pbar = tqdm.tqdm(total = N)
        while len(ts) < N and len(idxs) > 0:
            i = np.random.choice(len(idxs))

            # try:
            #     df = self.load_hdf(self.files[idxs[i]], mode=mode)
            # except Exception:
            #     print(len(idxs))
            #     print(i)
            #     print(len(self.files))

            z_df, m_df = self.load_hdf(self.files[idxs[i]], mode=mode)

            if len(z_df) < 10000:#T:
                mask = np.ones(len(idxs), dtype=bool)
                mask[i] = 0
                idxs = idxs[mask]
                continue

            cartesian_names = ['_'.join([var, str(p)]) for var in 'x y z vx vy vz'.split(' ') for p in range(3)]
            if np.any(np.abs(z_df[cartesian_names].to_numpy()) > 100):            
                mask = np.ones(len(idxs), dtype=bool)
                mask[i] = 0
                idxs = idxs[mask]
                continue

            t0 = np.random.choice(len(z_df) - T)
            z_df = z_df.iloc[t0:t0 + T]
            m_df = m_df.iloc[t0]
            
            # print(z_df.t.values)

            ts.append(z_df.t.values - z_df.t.values[0])
            zs.append(z_df.drop(columns=['t']).values)
            ms.append(m_df.values)

            # print(np.mean(m_df.values, 0))

            mask = np.ones(len(idxs), dtype=bool)
            mask[i] = 0
            idxs = idxs[mask]

            pbar.update(1)

            # for t0 in range(0, len(df), 5*T):
            #     _df = df.iloc[t0:t0 + T]
            #     ts.append(_df.t.values)
            #     zs.append(_df.drop(columns=['t']).values)

            #     pbar.update(1)

        print(np.c_[zs].shape)

        ts = np.c_[ts]
        zs = np.c_[zs].reshape(len(zs), T, 2, -1)
        ms = np.c_[ms]

        return ts, zs, ms


def get_chaotic_eval_dataset(body, n_init=5, n_samples=10, eps_scale=1e-3, tau=10.0):
    eval_dir = os.path.join(os.environ['DATADIR'], "chnn")
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    eval_path = os.path.join(eval_dir, f"{body.n}_body_eval_Ninit-{n_init}_Ns-{n_samples}.pt")
    if False:#os.path.exists(eval_path):
        return torch.load(eval_path)

    with FixedSeedAll(0):
        z0_orig = body.sample_initial_conditions(n_init)

        z0_orig_dup = z0_orig.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        z0_orig_dup = z0_orig_dup.reshape(n_init * n_samples, *np.shape(z0_orig_dup)[2:])
        eps = torch.rand_like(z0_orig_dup) #2. * torch.rand_like(z0_orig_dup) - 1.
        z0 = z0_orig_dup + eps_scale * eps
        z0 = project_onto_constraints(body.body_graph, z0, tol=1e-5)
    
    ts = torch.arange(0., tau, body.dt, device=z0_orig.device, dtype=z0_orig.dtype)

    true_zt = body.integrate(z0_orig, ts, method='rk4')
    true_zt_chaos = body.integrate(z0, ts, method='rk4')
    true_zt_chaos = true_zt_chaos.reshape(n_init, n_samples, *np.shape(true_zt_chaos)[1:])
    true_zt_chaos = true_zt_chaos.permute(1, 0, 2, 3, 4, 5)

    eval_dataset = {
        "ts": ts,
        "z0_orig": z0_orig,
        "true_zt": true_zt,
        "true_zt_chaos": true_zt_chaos
    }

    torch.save(eval_dataset, eval_path)

    return eval_dataset


def pendulum_near_separatrix(seed=0, amount_outside_separatrix=1e-3):#body, 
    body = ChainPendulum(1)
    n = len(body.body_graph.nodes)
    tau = 10
    angles_and_angvel = torch.zeros(1, 2, n)
    angles_and_angvel[0, 0, 0] = np.pi
    angles_and_angvel[0, 1, 0] = 0.0
    z = body.body2globalCoords(angles_and_angvel)
    separatrix_energy = body.total_energy(z)[0]
    with FixedSeedAll(seed):
        z_orig = body.sample_initial_conditions(1)
        outside_separatrix = 2 * (np.random.rand() > 0.5) - 1
        angles_and_angvel = np.random.randn(1, 2, n)  # (N,2,n)
        def adjust_vel(v):
            ccoord = angles_and_angvel.copy()
            ccoord[0, 1, 0] = v
            ccoord = torch.tensor(ccoord)
            z = body.body2globalCoords(ccoord)
            return (body.total_energy(z.float()) - separatrix_energy)**2
        res = minimize_scalar(adjust_vel)
        exactly_at_separatrix = res.x
        angles_and_angvel[0, 1, 0] = (
            exactly_at_separatrix
            + (np.sign(res.x)
               *outside_separatrix
               *amount_outside_separatrix)
        )
    z = body.body2globalCoords(torch.tensor(angles_and_angvel))
    z0_orig = z.float()
    ts = torch.arange(0., tau, body.dt,
              device=z0_orig.device,
              dtype=z0_orig.dtype)
    true_zt = body.integrate(z0_orig, ts, method='rk4')
    out = {
	"ts": ts,
	"z0_orig": z0_orig,
	"true_zt": true_zt,
	"body": body
    }
    return out
