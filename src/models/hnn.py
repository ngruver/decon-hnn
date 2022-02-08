import sys
import torch
from torch import Tensor
import torch.nn as nn

import numpy as np
from torchdiffeq import odeint
from .utils import FCsoftplus, FCtanh, FCswish, Reshape, Linear, CosSin
from ..dynamics.hamiltonian import HamiltonianDynamics, mHamiltonianDynamics, GeneralizedT
from typing import Tuple

def get_linear_layers(net):
    layers = []
    for layer in net:
        if isinstance(layer, nn.Linear):
            layers.append(layer)
        elif isinstance(layer, nn.Sequential):
            layers += [_layer for _layer in layer \
                        if isinstance(_layer, nn.Linear)]
    return layers

class HNN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        canonical: bool = False,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__()
        self.nfe = 0
        self.canonical = canonical

        self.q_ndim = dof_ndim
        self.angular_dims = angular_dims

        # We parameterize angular dims in terms of cos(theta), sin(theta)
        chs = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        self.potential_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], 1, zero_bias=False, orthogonal_init=True),
            Reshape(-1)
        )
        print("HNN currently assumes potential energy depends only on q")
        print("HNN currently assumes time independent Hamiltonian")

        self.mass_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], self.q_ndim * self.q_ndim, zero_bias=True, orthogonal_init=True),
            Reshape(-1, self.q_ndim, self.q_ndim)
        )

        self.dynamics = HamiltonianDynamics(self.H, wgrad=wgrad)

        for layer in get_linear_layers(self.potential_net):
            layer.weight.data = layer.weight.data

    def H(self, t, z):
        """ Compute the Hamiltonian H(t, q, p)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [q, p].
        Returns: Size N Hamiltonian Tensor
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim
        q, p = z.chunk(2, dim=-1)

        V = self.potential_net(q)

        Minv = self.Minv(q)
        T = GeneralizedT(p, Minv)
        return T + V

    def tril_Minv(self, q):
        mass_net_q = self.mass_net(q)
        res = torch.triu(mass_net_q, diagonal=1)
        # Constrain diagonal of Cholesky to be positive
        res = res + torch.diag_embed(
            torch.nn.functional.softplus(torch.diagonal(mass_net_q, dim1=-2, dim2=-1)),
            dim1=-2,
            dim2=-1,
        )
        #print(torch.nn.functional.softplus(torch.diagonal(mass_net_q, dim1=-2, dim2=-1)).min())
        res = res.transpose(-1, -2)  # Make lower triangular
        return res

    def Minv(self, q: Tensor, eps=1e-4) -> Tensor:
        """Compute the learned inverse mass matrix M^{-1}(q)
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q)
        assert lower_triangular.ndim == 3
        diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=q.dtype, device=q.device)
        Minv = lower_triangular.matmul(lower_triangular.transpose(-2, -1)) + diag_noise
        # print(torch.symeig(Minv)[0].mean(0))
        return Minv

    def M(self, q, eps=1e-4):
        """Returns a function that multiplies the mass matrix M(q) by a vector qdot
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q)
        assert lower_triangular.ndim == 3

        def M_func(qdot):
            assert qdot.ndim == 2
            qdot = qdot.unsqueeze(-1)
            diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=qdot.dtype, device=qdot.device)
            # print(f"mass matrix: {(lower_triangular @ lower_triangular.transpose(-2, -1) + diag_noise).mean()}")
            Minv = lower_triangular @ lower_triangular.transpose(-2, -1) + diag_noise
            # print(Minv.mean(0))
            # print(torch.symeig(Minv)[0].mean(0))

            M_times_qdot = torch.solve(qdot, Minv).solution.squeeze(-1)
            return M_times_qdot

        return M_func

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        Returns: N x D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        dz_dt = self.dynamics(t, z)
        # dz_dt[z.abs() > 1] = 0
        self.nfe += 1
        return dz_dt

    def integrate(self, z0, ts, tol=1e-5, method="rk4"):
        """ Integrates an initial state forward in time according to the learned Hamiltonian dynamics
        Args:
            z0: (N x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance
        Returns: a N x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        assert z0.shape[-1] == self.q_ndim
        bs, _, D = z0.size()
        assert D == self.q_ndim
        z0 = z0.reshape(bs, -1)  # -> bs x D
        if self.canonical:
            q0, p0 = z0.chunk(2, dim=-1)
        else:
            q0, v0 = z0.chunk(2, dim=-1)
            p0 = self.M(q0)(v0) #(DxD)*(bsxD) -> (bsxD)

        self.nfe = 0  # reset each forward pass
        qp0 = torch.cat([q0, p0], dim=-1)
        qpt = odeint(self, qp0, ts, rtol=tol, method=method)
        qpt = qpt.permute(1, 0, 2)  # T x N x D -> N x T x D

        # print(f"mean: {qpt.mean((0,1)).cpu().detach().numpy()}")
        # print(torch.min(qpt.reshape(-1, qpt.shape[-1])))
        # print(torch.max(qpt.reshape(-1, qpt.shape[-1])))
        # print("\n")

        # self._acc_magn = self.acc_magn(qpt)
        # self._H_magn = self.H_magn(qpt)

        if self.canonical:
            qpt = qpt.reshape(bs, len(ts), 2, D)
            return qpt
        else:
            qt, pt = qpt.reshape(-1, 2 * self.q_ndim).chunk(2, dim=-1)
            vt = self.Minv(qt).matmul(pt.unsqueeze(-1)).squeeze(-1)
            qvt = torch.cat([qt, vt], dim=-1)
            qvt = qvt.reshape(bs, len(ts), 2, D)
            return qvt

    def acc_magn(self, qpt):
        dz_dt = self.dynamics(torch.zeros(1)[0], qpt.reshape(-1, qpt.shape[-1]))
        magnitude = dz_dt.chunk(2, dim=-1)[1].pow(2).mean()
        return magnitude

    def H_magn(self, qpt):
        H = self.H(torch.zeros(1)[0], qpt.reshape(-1, qpt.shape[-1]))
        magnitude = H.pow(2).mean()
        return magnitude

    @property
    def param_groups(self):
        return self.parameters()

    # def log_data(self,logger,step,name):
    #     logger.add_scalars('info',
    #                        {'acc_magn': self._acc_magn.cpu().data.numpy(),
    #                         'H_magn': self._H_magn.cpu().data.numpy()}, 
    #                        step)


class NonseparableHNN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        canonical: bool = False,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nfe = 0
        self.canonical = canonical

        self.q_ndim = dof_ndim
        self.angular_dims = angular_dims

        # We parameterize angular dims in terms of cos(theta), sin(theta)
        chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        self.h_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], 1, zero_bias=False, orthogonal_init=True),
            Reshape(-1)
        )
        print("HNN currently assumes potential energy depends only on q")
        print("HNN currently assumes time independent Hamiltonian")

        chs = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        self.mass_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], self.q_ndim * self.q_ndim, zero_bias=True, orthogonal_init=True),
            Reshape(-1, self.q_ndim, self.q_ndim)
        )

        self.dynamics = HamiltonianDynamics(self.H, wgrad=wgrad)

    def H(self, t, z):
        """ Compute the Hamiltonian H(t, q, p)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [q, p].
        Returns: Size N Hamiltonian Tensor
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim

        return self.h_net(z)

    def tril_Minv(self, q):
        mass_net_q = self.mass_net(q)
        res = torch.triu(mass_net_q, diagonal=1)
        # Constrain diagonal of Cholesky to be positive
        res = res + torch.diag_embed(
            torch.nn.functional.softplus(torch.diagonal(mass_net_q, dim1=-2, dim2=-1)),
            dim1=-2,
            dim2=-1,
        )
        #print(torch.nn.functional.softplus(torch.diagonal(mass_net_q, dim1=-2, dim2=-1)).min())
        res = res.transpose(-1, -2)  # Make lower triangular
        return res

    def Minv(self, q: Tensor, eps=1e-4) -> Tensor:
        """Compute the learned inverse mass matrix M^{-1}(q)
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q)
        assert lower_triangular.ndim == 3
        diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=q.dtype, device=q.device)
        Minv = lower_triangular.matmul(lower_triangular.transpose(-2, -1)) + diag_noise
        # print(Minv.mean(0))
        # print(torch.symeig(Minv)[0].mean(0))
        return Minv

    def M(self, q, eps=1e-4):
        """Returns a function that multiplies the mass matrix M(q) by a vector qdot
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q)
        assert lower_triangular.ndim == 3

        def M_func(qdot):
            assert qdot.ndim == 2
            qdot = qdot.unsqueeze(-1)
            diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=qdot.dtype, device=qdot.device)
            M_times_qdot = torch.solve(
                    qdot,
                    lower_triangular @ lower_triangular.transpose(-2, -1) + diag_noise
            ).solution.squeeze(-1)
            return M_times_qdot

        return M_func

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        Returns: N x D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        dz_dt = self.dynamics(t, z)
        self.nfe += 1
        return dz_dt

    def integrate(self, z0, ts, tol=1e-5, method="rk4"):
        """ Integrates an initial state forward in time according to the learned Hamiltonian dynamics
        Args:
            z0: (N x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance
        Returns: a N x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        assert z0.shape[-1] == self.q_ndim
        bs, _, D = z0.size()
        assert D == self.q_ndim
        z0 = z0.reshape(bs, -1)  # -> bs x D
        if self.canonical:
            q0, p0 = z0.chunk(2, dim=-1)
        else:
            q0, v0 = z0.chunk(2, dim=-1)
            p0 = self.M(q0)(v0) #(DxD)*(bsxD) -> (bsxD)

        self.nfe = 0  # reset each forward pass
        qp0 = torch.cat([q0, p0], dim=-1)
        qpt = odeint(self, qp0, ts, rtol=tol, method=method)
        qpt = qpt.permute(1, 0, 2)  # T x N x D -> N x T x D

        if self.canonical:
            qpt = qpt.reshape(bs, len(ts), 2, D)
            return qpt
        else:
            qt, pt = qpt.reshape(-1, 2 * self.q_ndim).chunk(2, dim=-1)
            vt = self.Minv(qt).matmul(pt.unsqueeze(-1)).squeeze(-1)
            qvt = torch.cat([qt, vt], dim=-1)
            qvt = qvt.reshape(bs, len(ts), 2, D)
            return qvt

