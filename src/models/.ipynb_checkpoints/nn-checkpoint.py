import sys
import torch
import torch.nn as nn
from torchdiffeq import odeint
from .utils import FCsoftplus,FCtanh, Linear, CosSin
from typing import Tuple, Union
from ..uncertainty.swag import SWAG

class NN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if wgrad:
            print("NN ignores wgrad")
        self.q_ndim = dof_ndim

        # We parameterize angular dims in terms of cos(theta), sin(theta)
        chs = [2 * self.q_ndim + len(angular_dims)] + num_layers * [hidden_size]
        
        layers = [CosSin(self.q_ndim, angular_dims, only_q=False)] + \
                 [FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                    for i in range(num_layers)] + \
                 [Linear(chs[-1], 2 * self.q_ndim, zero_bias=False, orthogonal_init=True)]   

        self.net = nn.Sequential(*layers)

        wrap = lambda: nn.Sequential(*layers)
        self.swag_model = SWAG(wrap)

        print("NN currently assumes time independent ODE")
        self.nfe = 0
        self.angular_dims = angular_dims

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x 2D Tensor of the N different states in D dimensions
        Returns: N x 2D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim
        self.nfe += 1
        return self.net(z)

    def _integrate(self, dynamics, z0, ts, tol=1e-4, method="rk4"):
        """ Integrates an initial state forward in time according to the learned dynamics
        Args:
            z0: (bs x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance
        Returns: a bs x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        bs = z0.shape[0]
        self.nfe = 0
        zt = odeint(dynamics, z0.reshape(bs, -1), ts, rtol=tol, method=method)
        zt = zt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return zt.reshape(bs, len(ts), *z0.shape[1:])    

    def integrate(self, z0, ts, tol=1e-4, method="rk4"):
        """ Integrates an initial state forward in time according to the learned dynamics
        Args:
            z0: (bs x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance
        Returns: a bs x T x 2 x D sized Tensor
        """
        return self._integrate(lambda t,z: self.forward(t,z), z0, ts, tol, method)

    def integrate_swag(self, z0, ts, tol=1e-4, method="rk4"):
        return self._integrate(lambda t, z: self.swag_model(z), z0, ts, tol, method)

    def collect_model(self):
        self.swag_model.collect_model(self.net)

    def sample(self):
        self.swag_model.sample()


class mNN(nn.Module):
    def __init__(
        self,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if wgrad:
            print("NN ignores wgrad")
        self.q_ndim = dof_ndim

        self.cossin = CosSin(3 * self.q_ndim, angular_dims, only_q=False)

        # We parameterize angular dims in terms of cos(theta), sin(theta)
        chs = [3 * self.q_ndim  + len(angular_dims)] + num_layers * [hidden_size]
        
        layers = [CosSin(2 * self.q_ndim, angular_dims, only_q=False)] + \
                 [FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                    for i in range(num_layers)] + \
                 [Linear(chs[-1], 2 * self.q_ndim, zero_bias=False, orthogonal_init=True)]   

        self.net = nn.Sequential(*layers)

#         wrap = lambda: nn.Sequential(*layers)
#         self.swag_model = SWAG(wrap)

        print("NN currently assumes time independent ODE")
        self.nfe = 0
        self.angular_dims = angular_dims

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x 2D Tensor of the N different states in D dimensions
        Returns: N x 2D Tensor of the time derivatives
        """
        z, m = z
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim
        self.nfe += 1
        zm = torch.cat([z, m], dim=1)
        dz = self.net(zm)
#         if self.training:
#             dz[:,:self.q_ndim] = dz[:,:self.q_ndim] + 0.01 * torch.randn_like(dz[:,:self.q_ndim])
        dm = torch.zeros_like(m)
        return dz, dm  

    def integrate(self, z0, m, ts, tol=1e-4, method="rk4"):
        """ Integrates an initial state forward in time according to the learned dynamics
        Args:
            z0: (bs x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance
        Returns: a bs x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        bs = z0.shape[0]
        self.nfe = 0
        zt, _ = odeint(self, (z0.reshape(bs, -1), m), ts, rtol=tol, method=method)
        zt = zt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return zt.reshape(bs, len(ts), *z0.shape[1:])    

class ControlNN(nn.Module):
    def __init__(
        self,
        control_policy,
        G,
        dof_ndim: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        angular_dims: Tuple = tuple(),
        wgrad: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if wgrad:
            print("NN ignores wgrad")
        self.q_ndim = dof_ndim

        # We parameterize angular dims in terms of cos(theta), sin(theta)
        chs = [2 * self.q_ndim + len(angular_dims) + 1] + num_layers * [hidden_size]
        
        layers = [CosSin(self.q_ndim, angular_dims, only_q=False)] + \
                 [FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                    for i in range(num_layers)] + \
                 [Linear(chs[-1], 2 * self.q_ndim, zero_bias=False, orthogonal_init=True)]   

        self.net = nn.Sequential(*layers)

        chs = [1] + num_layers * [hidden_size]
        
        layers = [FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                    for i in range(num_layers)] + \
                 [Linear(chs[-1], 2 * self.q_ndim, zero_bias=False, orthogonal_init=True)]   

        self.control_net = nn.Sequential(*layers)

        print("NN currently assumes time independent ODE")
        self.nfe = 0
        self.angular_dims = angular_dims

        self.control_policy = control_policy

    def forward(self, t, z):
        """ Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: N x 2D Tensor of the N different states in D dimensions
        Returns: N x 2D Tensor of the time derivatives
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.q_ndim
        self.nfe += 1
        u = self.control_policy(t, z).detach()
        # print(u)
        dynamics = self.net(torch.cat([z,u], axis=-1)) #self.net(z) + self.control_net(u)
        # print(dynamics)
        return dynamics

    def _integrate(self, dynamics, z0, ts, tol=1e-4, method="rk4"):
        """ Integrates an initial state forward in time according to the learned dynamics
        Args:
            z0: (bs x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance
        Returns: a bs x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        bs = z0.shape[0]
        self.nfe = 0
        zt = odeint(dynamics, z0.reshape(bs, -1), ts, rtol=tol, method=method)
        zt = zt.permute(1, 0, 2)  # T x N x D -> N x T x D
        return zt.reshape(bs, len(ts), *z0.shape[1:])    

    def integrate(self, z0, ts, tol=1e-4, method="rk4"):
        """ Integrates an initial state forward in time according to the learned dynamics
        Args:
            z0: (bs x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance
        Returns: a bs x T x 2 x D sized Tensor
        """
        return self._integrate(lambda t,z: self.forward(t,z), z0, ts, tol, method)

    def integrate_swag(self, z0, ts, tol=1e-4, method="rk4"):
        return self._integrate(lambda t, z: self.swag_model(z), z0, ts, tol, method)

    def collect_model(self):
        self.swag_model.collect_model(self.net)

    def sample(self):
        self.swag_model.sample()



class DeltaNN(NN):
    def integrate(self, z0, ts, tol=0.0,method=None):
        """ Integrates an initial state forward in time according to the learned
        dynamics using Euler's method with predicted time derivatives
        Args:
            z0: (bs x 2 x D) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
        Returns: a bs x T x 2 x D sized Tensor
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        bs = z0.shape[0]
        dts = ts[1:] - ts[:-1]
        zts = [z0.reshape(bs, -1)]
        for dt in dts:
            zts.append(zts[-1] + dt * self(ts[0], zts[-1]))
        return torch.stack(zts, dim=1).reshape(bs, len(ts), *z0.shape[1:])
