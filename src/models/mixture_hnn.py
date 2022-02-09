import sys
import torch
from torch import Tensor
import torch.nn as nn

from torchdiffeq import odeint
from .nn import NN
from .hnn import HNN
from .utils import convert_linear_to_snlinear
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

class MixtureHNN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        action_dim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        canonical: bool = False,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        alpha: float = 1e8,
        beta: float = 1e8,
        tie_layers: int = 0,
        spectral_norm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.verbose = False

        self.nfe = 0
        self.canonical = canonical

        self.q_ndim = dof_ndim
        self.angular_dims = angular_dims

        self.alpha = alpha
        self.beta = beta
        self.alpha_ratio = self.alpha / (self.alpha + self.beta)

        self.nn = NN(G, dof_ndim, hidden_size, num_layers, angular_dims, wgrad, **kwargs)
        self.hnn = HNN(G, dof_ndim, hidden_size, num_layers, canonical, angular_dims, wgrad, **kwargs)

        self.tie_layers = tie_layers

        self.do_initialization()

        self.spectral_norm = spectral_norm
        if self.spectral_norm:
            convert_linear_to_snlinear(self)

        if self.verbose:
            for i, (nn_layer, hnn_layer) in enumerate(zip(self.nn.net, self.hnn.h_net)):
                if isinstance(nn_layer, nn.Linear) and isinstance(hnn_layer, nn.Linear):
                    print(nn_layer.weight[0][:10])
                    print(hnn_layer.weight[0][:10])

                if isinstance(nn_layer, nn.Sequential) and isinstance(hnn_layer, nn.Sequential):
                    for j, (_nn_layer, _hnn_layer) in enumerate(zip(nn_layer, hnn_layer)):
                        if isinstance(_nn_layer, nn.Linear) and isinstance(_hnn_layer, nn.Linear):
                            print(_nn_layer.weight[0][:10])
                            print(_hnn_layer.weight[0][:10])

    def do_initialization(self):
        self.mass_net = self.hnn.mass_net

        if not self.tie_layers:
            return

        num_tied = 0
        for i, (nn_layer, hnn_layer) in enumerate(zip(self.nn.net, self.hnn.h_net)):
            if isinstance(nn_layer, nn.Linear) and isinstance(hnn_layer, nn.Linear):
                self.nn.net[i] = hnn_layer
                num_tied += 1

            if isinstance(nn_layer, nn.Sequential) and isinstance(hnn_layer, nn.Sequential):
                for j, (_nn_layer, _hnn_layer) in enumerate(zip(nn_layer, hnn_layer)):
                    if isinstance(_nn_layer, nn.Linear) and isinstance(_hnn_layer, nn.Linear):
                        self.nn.net[i][j] = _hnn_layer
                        num_tied += 1

            if num_tied == self.tie_layers:
                break

    def tril_Minv(self, q):
        mass_net_q = self.mass_net(q)
        res = torch.triu(mass_net_q, diagonal=1)
        # Constrain diagonal of Cholesky to be positive
        res = res + torch.diag_embed(
            torch.nn.functional.softplus(torch.diagonal(mass_net_q, dim1=-2, dim2=-1)),
            dim1=-2,
            dim2=-1,
        )
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

    def J(self, M):
        """ applies the J matrix to another matrix M.
            input: M (*,2nd,b), output: J@M (*,2nd,b)"""
        *star, D, b = M.shape
        JM = torch.cat([M[..., D // 2 :, :], -M[..., : D // 2, :]], dim=-2)
        return JM

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x D Tensor of the N different states in D dimensions
        Returns: N x D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)

        hnn_dyn = self.hnn(t, z)
        nn_dyn = self.nn(t, z)
        dz_dt = hnn_dyn + nn_dyn

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
