import sys
import torch
from torch import Tensor
import torch.nn as nn

import numpy as np
from torchdiffeq import odeint
from .utils import FCsoftplus, FCtanh, FCswish, Reshape, Linear, CosSin
from ..dynamics.hamiltonian import HamiltonianDynamics, mHamiltonianDynamics, GeneralizedT
from typing import Tuple
from torch.autograd.functional import jacobian
from .nn import NN

def get_linear_layers(net):
    layers = []
    for layer in net:
        if isinstance(layer, nn.Linear):
            layers.append(layer)
        elif isinstance(layer, nn.Sequential):
            layers += [_layer for _layer in layer \
                        if isinstance(_layer, nn.Linear)]
    return layers

class SpNN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        canonical: bool = False,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        alpha=1e3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nfe = 0
        self.canonical = canonical

        self.q_ndim = dof_ndim
        self.angular_dims = angular_dims

        # We parameterize angular dims in terms of cos(theta), sin(theta)
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
        chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        self.dynamics_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], self.q_ndim, zero_bias=False, orthogonal_init=True),
            Reshape(-1, self.q_ndim)
        )

        # self.dynamics_net = NN(G, dof_ndim, hidden_size, num_layers, angular_dims, wgrad, **kwargs)
        self.alpha=alpha

    def tril_Minv(self, q):
        mass_net_q = self.mass_net(q)
        # print(mass_net_q.mean(0))
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
        return Minv

    def M(self, q, eps=1e-6):
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
        assert (z.ndim == 2)
        q, p = z.chunk(2, dim=-1)
        dq_dt = self.Minv(q).matmul(p.unsqueeze(-1)).squeeze(-1)
        dp_dt = self.dynamics_net(z)
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)
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
        # Evaluate regularization term on each timestep
        self._regularization = self.symplectic_prior_reg(qpt)
        if self.canonical:
            qpt = qpt.reshape(bs, len(ts), 2, D)
            return qpt
        else:
            qt, pt = qpt.reshape(-1, 2 * self.q_ndim).chunk(2, dim=-1)
            vt = self.Minv(qt).matmul(pt.unsqueeze(-1)).squeeze(-1)
            qvt = torch.cat([qt, vt], dim=-1)
            qvt = qvt.reshape(bs, len(ts), 2, D)
            return qvt

    def J(self, M):
        """ applies the J matrix to another matrix M.
            input: M (*,2nd,b), output: J@M (*,2nd,b)"""
        *star, D, b = M.shape
        JM = torch.cat([M[..., D // 2 :, :], -M[..., : D // 2, :]], dim=-2)
        return JM

    def symplectic_prior_reg(self,z):
        """ Computes the symplectic prior regularization term
        Args:
            z: N x T x D Tensor representing the state
        Returns: ||(JDF)^T - JDF||^2
        """
        with torch.enable_grad():
            D = z.shape[-1]
            F = lambda z: self.forward(None, z).sum(0)
            DF = jacobian(F, z.reshape(-1,D), create_graph=True, vectorize=True) # (D,NT,D)
            JDF = self.J(DF.permute(1,0,2)) # (NT,D,D)
            reg = (JDF-JDF.transpose(-1,-2)).pow(2).mean()
            return reg

    def regularization(self, trainer=None):
        """ Computes the regularization term
        Returns: ||(JDF)^T - JDF||^2
        """
        return self._regularization/self.alpha

    def log_data(self,logger,step,name):
        logger.add_scalars('info',{'reg':self._regularization.cpu().data.numpy()}, step)


class MechanicsNN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        canonical: bool = False,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        alpha=1e8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nfe = 0
        self.canonical = canonical

        self.q_ndim = dof_ndim
        self.angular_dims = angular_dims

        # We parameterize angular dims in terms of cos(theta), sin(theta)
        chs = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        self.mass_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            # *[val for pair in zip([FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
            #         for i in range(num_layers)],
            #      [nn.Dropout(0.2) for i in range(num_layers)])
            #   for val in pair
            # ],
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], self.q_ndim * self.q_ndim, zero_bias=True, orthogonal_init=True),
            Reshape(-1, self.q_ndim, self.q_ndim)
        )

        chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        self.dynamics_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            # *[val for pair in zip([FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
            #         for i in range(num_layers)],
            #      [nn.Dropout(0.5) for i in range(num_layers)])
            #   for val in pair
            # ],
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], self.q_ndim, zero_bias=False, orthogonal_init=True),
            Reshape(-1, self.q_ndim)
        )

        self.alpha=alpha

    def tril_Minv(self, q):
        mass_net_q = self.mass_net(q)
        # print(mass_net_q.mean(0))
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
        return Minv

    def M(self, q, eps=1e-6):
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
            # print(M_times_qdot.mean())
            return M_times_qdot

        return M_func

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        Returns: N x D Tensor of the time derivatives
        """
        assert (z.ndim == 2)
        q, p = z.chunk(2, dim=-1)
        dq_dt = self.Minv(q).matmul(p.unsqueeze(-1)).squeeze(-1)
        dp_dt = self.dynamics_net(z)
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)
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
        
        self._acc_magn = self.acc_magn(qpt)

        # Evaluate regularization term on each timestep
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
        dz_dt = self.forward(torch.zeros(1)[0], qpt.reshape(-1, qpt.shape[-1]))
        magnitude = dz_dt.chunk(2, dim=-1)[1].pow(2).mean()
        return magnitude

    def log_data(self,logger,step,name):
        logger.add_scalars('info',
                           {'acc_magn': self._acc_magn.cpu().data.numpy()}, 
                           step)

class SecondOrderNN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        canonical: bool = False,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        alpha=1e8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nfe = 0
        self.canonical = canonical

        self.q_ndim = dof_ndim
        self.angular_dims = angular_dims

        chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        self.dynamics_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            *[
                FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)
            ],
            Linear(chs[-1], 2 * self.q_ndim, zero_bias=False, orthogonal_init=True),
            Reshape(-1, 2 * self.q_ndim)
        )

        self.scale = nn.Parameter(torch.ones(self.q_ndim))
        self.shift = nn.Parameter(torch.zeros(self.q_ndim))

        self.alpha=alpha


    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        Returns: N x D Tensor of the time derivatives
        """
        assert (z.ndim == 2)
        v, p = z.chunk(2, dim=-1)
        dq_dt = v + self.dynamics_net(z).chunk(2, dim=-1)[0] #self.scale * v + self.shift
        dv_dt = self.dynamics_net(z).chunk(2, dim=-1)[1]
        dz_dt = torch.cat([dq_dt, dv_dt], dim=-1)
        self.nfe += 1
        return dz_dt


    def integrate(self, z0, ts, tol=1e-5, method="euler"):
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

        self.nfe = 0  # reset each forward pass

        ts = ts - ts[0]
        # print(ts)

        z0 = z0.reshape(bs, -1)  # -> bs x D
        zt = odeint(self, z0, ts, rtol=tol, method=method)
        zt = zt.permute(1, 0, 2)  # T x N x D -> N x T x D
        zt = zt.reshape(bs, len(ts), 2, D)

        # print(f"scale: {self.scale.data}")
        # print(f"shift: {self.shift.data}")
        # print("\n")

        return zt
