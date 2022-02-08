import sys
import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd.functional import jacobian

import numpy as np
from torchdiffeq import odeint
from .nn import NN
from .hnn import HNN, NonseparableHNN
from .utils import FCsoftplus, FCtanh, FCswish, Reshape, Linear, CosSin
from .utils import convert_linear_to_snlinear
from ..dynamics.hamiltonian import HamiltonianDynamics, mHamiltonianDynamics, GeneralizedT
from typing import Tuple

class FlexHNN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        canonical: bool = False,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        alpha: float = 1e5,
        beta: float = 1e5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nfe = 0
        self.canonical = canonical

        self.q_ndim = dof_ndim
        self.angular_dims = angular_dims

        self.alpha = alpha
        self.beta = beta
        self.alpha_ratio = self.alpha / (self.alpha + self.beta)

        activation = nn.Tanh

#         # We parameterize angular dims in terms of cos(theta), sin(theta)
        chs = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True) for i in range(num_layers)]
        activations = [activation() for i in range(num_layers)]
        self.mass_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], self.q_ndim * self.q_ndim, zero_bias=True, orthogonal_init=True),
            Reshape(-1, self.q_ndim, self.q_ndim)
        )
        
        chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True) for i in range(num_layers)]
        activations = [activation() for i in range(num_layers)]
        self.h_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], 1, zero_bias=False, orthogonal_init=True),
            Reshape(-1)
        )
        
        chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True) for i in range(num_layers)]
        activations = [activation() for i in range(num_layers)]
        self.flex_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], 2 * self.q_ndim + len(angular_dims), zero_bias=False, orthogonal_init=True),
            Reshape(-1, 2 * self.q_ndim + len(angular_dims))
        )

        print("H net")
        print(self.h_net)
        print("")
        print("Flex net")
        print(self.flex_net)
        print("\n")

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
        
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z

            out = self.h_net[0](z)
            sps = []
            for i, h_layer in enumerate(self.h_net[1:-1]):
                if i % 2 == 0:
                    out = h_layer(out)
                else:
                    d_activation = torch.autograd.grad(h_layer(out).sum(), out, create_graph=True)[0]
                    sps.append(d_activation)
                    out = h_layer(out)

            dH = torch.autograd.grad(out.sum(), z, create_graph=True)[0]

            def wrap(z):
                return self.h_net[0](z).sum(0)

            D = jacobian(wrap, z, create_graph=True).transpose(0,1)

            out = self.flex_net[0](z)
            for i, flex_layer in enumerate(self.flex_net[1:-1]):
                if i % 2 == 0:
                    out = flex_layer(out)
                else:
                    idx = len(sps) - 1 - (i // 2)
                    # out = out * sps[idx] + flex_layer(out)
                    # The below line is necessary if we want to converge exactly
                    out = (1 - self.alpha_ratio) * out * sps[idx] + self.alpha_ratio * flex_layer(out)

            out = torch.bmm(out[:,None,:], D).squeeze(1)
            
            
            # YOU CAN CHECK HOW CLOSE THE OUTPUT OF THE NETWORK IS TO THE TRUE BACKWARDS PASS
            print(f"Diff: {(out - dH).pow(2).mean()}")

            dz_dt = self.J(out.unsqueeze(-1)).squeeze(-1)
            # dz_dt = self.J((out + dH).unsqueeze(-1)).squeeze(-1)

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

    def regularization(self, trainer=None):
        sim_loss = 0.
        reg_loss = 0.

        h_layers = self.h_net[1:-1][::-1]
        flex_layers = self.flex_net[1:-1]
        for i, (h_layer, flex_layer) in enumerate(zip(h_layers, flex_layers)):
            if i % 2 != 0:
                continue

            # YOU CAN USE THIS TO CHECK WHAT HAPPENS IF REGULARIZATION IS PERFECTLY EFFECTIVE
            # if i == 0:
            #     flex_layer.weight.data = torch.zeros_like(flex_layer.weight.data)
            #     flex_layer.bias = h_layer.weight
            # else:
            #     flex_layer.weight.data = h_layer.weight.data.t()
            #     flex_layer.bias.data = torch.zeros_like(flex_layer.bias)

            if i == 0:
                sim_loss += flex_layer.weight.pow(2).sum() / self.alpha #weight reg
                sim_loss += (h_layer.weight.t() - flex_layer.bias).pow(2).sum() / self.alpha #bias reg
                # sim_loss += (h_layer.weight.t().detach() - flex_layer.bias).pow(2).sum() / self.alpha #bias reg
                reg_loss += h_layer.weight.pow(2).sum() / self.beta
            else:
                sim_loss += flex_layer.bias.pow(2).sum() / self.alpha #bias reg
                sim_loss += (h_layer.weight.t() - flex_layer.weight).pow(2).sum() / self.alpha #weight reg
                # sim_loss += (h_layer.weight.t().detach() - flex_layer.weight).pow(2).sum() / self.alpha #weight reg
                reg_loss += h_layer.weight.pow(2).sum() / self.beta

        reg_loss = sim_loss + reg_loss
        return reg_loss

class ControlFlexHNN(nn.Module):
    def __init__(
        self,
        control_policy,
        G,
        dof_ndim: int = 1,
        action_dim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        canonical: bool = False,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        alpha: float = 1e5,
        beta: float = 1e5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nfe = 0
        self.canonical = canonical

        self.q_ndim = dof_ndim
        self.angular_dims = angular_dims

        self.alpha = alpha
        self.beta = beta
        self.alpha_ratio = self.alpha / (self.alpha + self.beta)

        activation = nn.Tanh

#         # We parameterize angular dims in terms of cos(theta), sin(theta)
        chs = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True) for i in range(num_layers)]
        activations = [activation() for i in range(num_layers)]
        self.mass_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], self.q_ndim * self.q_ndim, zero_bias=True, orthogonal_init=True),
            Reshape(-1, self.q_ndim, self.q_ndim)
        )
        
        chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True) for i in range(num_layers)]
        activations = [activation() for i in range(num_layers)]
        self.h_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], 1, zero_bias=False, orthogonal_init=True),
            Reshape(-1)
        )
        
        chs = [2 * self.q_ndim + len(angular_dims) + action_dim] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True) for i in range(num_layers)]
        activations = [activation() for i in range(num_layers)]
        self.flex_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], 2 * self.q_ndim + len(angular_dims), zero_bias=False, orthogonal_init=True),
            Reshape(-1, 2 * self.q_ndim + len(angular_dims))
        )
        

        print("H net")
        print(self.h_net)
        print("")
        print("Flex net")
        print(self.flex_net)
        print("\n")

        self.control_policy = control_policy


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
        
        # z = torch.zeros_like(z, requires_grad=True) + z
        with torch.enable_grad():
            z = torch.zeros_like(z, requires_grad=True) + z
            u = self.control_policy(t, z).detach()

            out = self.h_net[0](z)
            sps = []
            for i, h_layer in enumerate(self.h_net[1:-1]):
                if i % 2 == 0:
                    out = h_layer(out)
                else:
                    d_activation = torch.autograd.grad(h_layer(out).sum(), out, create_graph=True)[0]
                    sps.append(d_activation)
                    out = h_layer(out)

            dH = torch.autograd.grad(out.sum(), z, create_graph=True)[0]

            def wrap(z):
                return self.h_net[0](z).sum(0)

            D = jacobian(wrap, z, create_graph=True).transpose(0,1)

            out = self.flex_net[0](torch.cat([z, u], axis=-1))
            for i, flex_layer in enumerate(self.flex_net[1:-1]):
                if i % 2 == 0:
                    out = flex_layer(out)
                else:
                    idx = len(sps) - 1 - (i // 2)
                    # out = self.alpha_ratio * flex_layer(out) + (1 - self.alpha_ratio) * out * sps[idx]
                    out = flex_layer(out) + out * sps[idx]

            out = torch.bmm(out[:,None,:], D).squeeze(1)
            
            dz_dt = self.J((out + dH).unsqueeze(-1)).squeeze(-1)

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

    def regularization(self, trainer=None):
        reg_loss = 0.
        
        h_layers = self.h_net[1:-1][::-1]
        flex_layers = self.flex_net[1:-1]
        for i, (h_layer, flex_layer) in enumerate(zip(h_layers, flex_layers)):
            if i % 2 != 0:
                continue

            if i == 0:
                reg_loss += flex_layer.weight.pow(2).sum() / self.alpha #weight reg
                reg_loss += (h_layer.weight.t().detach() - flex_layer.bias).pow(2).sum() / self.alpha #bias reg
                reg_loss += h_layer.weight.pow(2).sum() / self.beta
            else:
                reg_loss += (h_layer.weight.t().detach() - flex_layer.weight).pow(2).sum() / self.alpha #weight reg
                reg_loss += h_layer.weight.pow(2).sum() / self.beta
                reg_loss += flex_layer.bias.pow(2).sum() / self.alpha #bias reg


        return reg_loss

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
        # if self.alpha < 1:
        #     for layer in get_linear_layers(self.nn.net):
        #         layer.weight.data = layer.weight.data * self.alpha
        # if self.beta < 1:
        #     for layer in get_linear_layers(self.hnn.h_net):
        #         layer.weight.data = layer.weight.data * self.beta

        # for layer in get_linear_layers(self.nn.net):
        #     layer.bias.data = layer.bias.data * 0

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
        # print(z.shape)
        assert (t.ndim == 0) and (z.ndim == 2)

        # nn_dyn = self.nn(t, z)

        # q, v = z.chunk(2, dim=-1)
        # qp = torch.cat([q, self.M(q)(v)], dim=-1)
        # dq_dt, dp_dt = self.hnn(t, qp).chunk(2, dim=-1)

        # M_func = lambda q: self.M(q)(dq_dt).sum(0)
        # dM_dq = jacobian(M_func, q, create_graph=True).permute(1,0,2)
        # dMv_dt = torch.matmul(dM_dq, dq_dt.unsqueeze(-1)).squeeze(-1)
        # dv_dt = self.Minv(q).matmul((dp_dt - dMv_dt).unsqueeze(-1)).squeeze(-1)

        # hnn_dyn = torch.cat([dq_dt, dv_dt], dim=-1)

        # if self.training:
        #     hnn_dyn = self.hnn(t, z)
        #     dz_dt = hnn_dyn
        #     if np.random.rand() < 0.5:
        #         nn_dyn = self.nn(t, z)
        #         dz_dt += nn_dyn

        # else:
        hnn_dyn = self.hnn(t, z)
        nn_dyn = self.nn(t, z)
        dz_dt = hnn_dyn + nn_dyn

        # print(f"Node: {nn_dyn.pow(2).mean()}")
        # print(f"HNN: {hnn_dyn.pow(2).mean()}")
        # print("")

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
        # print(z0.size())
        # print(self.q_ndim)
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

        # _z0 = z0.reshape(bs, -1)
        # zt = odeint(self, _z0, ts, rtol=tol, method=method)
        # zt = zt.reshape(len(ts), *z0.shape).permute(1, 0, 2, 3)

        # return zt

    def regularization(self, trainer=None):
        num_tied = 0
        reg_loss = 0.

        nn_layers = get_linear_layers(self.nn.net)
        hnn_layers = get_linear_layers(self.hnn.potential_net) #h_net)
        n_layers = len(nn_layers)

        for idx, (nn_layer, hnn_layer) in enumerate(zip(nn_layers, hnn_layers)):
            if num_tied < self.tie_layers:
                reg = max(self.alpha, self.beta)
                if not self.spectral_norm:
                    reg_loss += nn_layer.weight.pow(2).sum() / reg
                    reg_loss += hnn_layer.weight.pow(2).sum() / reg
                else:
                    reg_loss += nn_layer._s.pow(2).sum() / reg
                    reg_loss += hnn_layer._s.pow(2).sum() / reg
            else:     
                if not self.spectral_norm:           
                    reg_loss += nn_layer.weight.pow(2).sum() / self.alpha
                    reg_loss += hnn_layer.weight.pow(2).sum() / self.beta
                    if idx == (n_layers - 1):
                        reg_loss += nn_layer.bias.pow(2).sum() / self.alpha
                        reg_loss += hnn_layer.bias.pow(2).sum() / self.beta
                else:
                    reg_loss += nn_layer._s.pow(2).sum() / self.alpha
                    reg_loss += hnn_layer._s.pow(2).sum() / self.beta

            num_tied += 1

        # print(f"Regularization loss: {reg_loss}")
        return reg_loss
