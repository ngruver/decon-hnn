import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from ..models.utils import FCtanh,FCswish,FCsoftplus, Linear, Reshape
from ..dynamics.hamiltonian import (
    EuclideanT,
    ConstrainedHamiltonianDynamics,
)
from ..systems.rigid_body import rigid_DPhi,rigid_Phi
from typing import Optional, Tuple, Union
import networkx as nx
import torch.nn.functional as F
from ..uncertainty.swag import SWAG

def divergence_bf(dz, z, **unused_kwargs):
    sum_diag = 0.
    for i in range(z.shape[1]):
        grad = torch.autograd.grad(dz[:, i].sum(), z, create_graph=True)[0]
        sum_diag += grad.contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def standard_normal_logprob(z, std=1):
    logZ = -0.5 * math.log(2 * math.pi) - math.log(std)
    return logZ - (z / std).pow(2) / 2

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class CH(nn.Module):  # abstract constrained Hamiltonian network class
    def __init__(self,G,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        wgrad=True, **kwargs):

        super().__init__(**kwargs)
        if angular_dims != tuple():
            # print("CH ignores angular_dims")
            pass
        self.G = G
        self.nfe = 0
        self.wgrad = wgrad
        self.n_dof = len(G.nodes)
        self.dof_ndim = dof_ndim
        self.q_ndim = self.n_dof * self.dof_ndim
        self.dynamics = ConstrainedHamiltonianDynamics(self.H, self.DPhi, wgrad=self.wgrad)

        #self._Minv = torch.nn.Parameter(torch.eye(self.n_dof))
        # print("CH currently assumes potential energy depends only on q")
        # print("CH currently assumes time independent Hamiltonian")
        # print("CH assumes positions q are in Cartesian coordinates")
        #self.moments = torch.nn.Parameter(torch.ones(self.n_dof,self.n_dof))
        #self.masses = torch.nn.Parameter(torch.zeros(self.n_dof))
        #self.moments = torch.nn.Parameter(torch.zeros(self.dof_ndim,self.n_dof))
        
        moment_dict = {
            str(d): torch.nn.Parameter(.1*torch.randn(len(d_objs)//(d+1),d+1)) # N,d+1
                for d,d_objs in G.d2ids.items()
        }
        self.d_moments = nn.ParameterDict(moment_dict)

        wrap = lambda: nn.ParameterDict(moment_dict)
        self.swag_d_moments = SWAG(wrap)

    def Minv(self, p):
        """ assumes p shape (*,n,a) and n is organized, all the same dimension for now"""
        assert len(self.d_moments)==1, "For now only supporting 1 dimension at a time"
        d = int(list(self.d_moments.keys())[0])

        *start,n,a = p.shape
        N = n//(d+1) # number of extended bodies
        p_reshaped = p.reshape(*start,N,d+1,a) # (*, # separate bodies, # internal body nodes, a)
        inv_moments = torch.exp(-self.d_moments[str(d)])
        inv_masses = inv_moments[:,:1] # (N,1)
        if d==0: return (inv_masses.unsqueeze(-1)*p_reshaped).reshape(*p.shape)# no inertia for point masses
        padded_inertias_inv = torch.cat([0*inv_masses,inv_moments[:,1:]],dim=-1) # (N,d+1)
        inverse_massed_p = p_reshaped.sum(-2,keepdims=True)*inv_masses[:,:,None]
        total = inverse_massed_p + p_reshaped*padded_inertias_inv[:,:,None]
        return total.reshape(*p.shape)

    def M(self, v):
        """ assumes v has shape (*,n,a) and n is organized, all the same dimension for now"""
        assert len(self.d_moments)==1, "For now only supporting 1 dimension at a time"
        d = int(list(self.d_moments.keys())[0])
        *start,n,a = v.shape 
        N = n//(d+1) # number of extended bodies
        v_reshaped = v.reshape(*start,N,d+1,a) # (*, # separate bodies, # internal body nodes, a)       
        moments = torch.exp(self.d_moments[str(d)])
        masses = moments[:,:1]
        if d==0: return (masses.unsqueeze(-1)*v_reshaped).reshape(*v.shape) # no inertia for point masses
        a00 = (masses + moments[:,1:].sum(-1,keepdims=True)).unsqueeze(-1) #(N,1,1)
        ai0 = a0i = -moments[:,1:].unsqueeze(-1) #(N,d,1)
        p0 = a00*v[...,:1,:] + (a0i*v[...,1:,:]).sum(-2,keepdims=True)
        aii = moments[:,1:].unsqueeze(-1) # (N,d,1)
        
        pi = ai0*v[...,:1,:] +aii*v[...,1:,:]
        return torch.cat([p0,pi],dim=-2).reshape(*v.shape)

    def H(self, t, z):
        """ Compute the Hamiltonian H(t, x, p)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [x, p] where x is in Cartesian coordinates.
        Returns: Size N Hamiltonian Tensor
        """
       # assert (t.ndim == 0) and (z.ndim == 2)
        assert z.size(-1) == 2 * self.n_dof * self.dof_ndim

        x, p = z.chunk(2, dim=-1)
        x = x.reshape(-1, self.n_dof, self.dof_ndim)
        p = p.reshape(-1, self.n_dof, self.dof_ndim)

        T = EuclideanT(p, self.Minv)
        V = self.compute_V(x)
        return T + V

    def DPhi(self, zp):
        bs,n,d = zp.shape[0],self.n_dof,self.dof_ndim
        x,p = zp.reshape(bs,2,n,d).unbind(dim=1)
        v = self.Minv(p)
        DPhi = rigid_DPhi(self.G, x, v)
        # Convert d/dv to d/dp
        #DPhi[:,1] = 
        DPhi = torch.cat([DPhi[:,:1], self.Minv(DPhi[:,1].reshape(bs,n,-1)).reshape(DPhi[:,1:].shape)],dim=1)
        return DPhi.reshape(bs,2*n*d,-1)

    def Phi(self, zp):
        bs,n,d = zp.shape[0],self.n_dof,self.dof_ndim
        x,p = zp.reshape(bs,2,n,d).unbind(dim=1)
        v = self.Minv(p)
        Phi = rigid_Phi(self.G, x, v)
        return Phi.reshape(bs,-1)

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def compute_V(self, x):
        raise NotImplementedError

    def to_pos_momentum(self, z0, models=None):
        bs = z0.size(0)
        x0, xdot0 = z0.chunk(2, dim=1)

        if models is None:
            p0 = self.M(xdot0)
        else:
            p0 = []
            for model in models:
                p0.append(model.M(xdot0))
            p0 = torch.stack(p0, 0).mean(0)

        xp0 = torch.stack([x0, p0], dim=1).reshape(bs,-1)
        return xp0

    def integrate(self, z0, ts, tol=1e-4, method="rk4", w_div=False, models=None):
        """ Integrates an initial state forward in time according to the learned Hamiltonian dynamics
        Assumes that z0 = [x0, xdot0] where x0 is in Cartesian coordinates
        Args:
            z0: (N x 2 x n_dof x dof_ndim) sized
                Tensor representing initial state. N is the batch size
            ts: a length T Tensor representing the time points to evaluate at
            tol: integrator tolerance
        Returns: a N x T x 2 x n_dof x d sized Tensor
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        assert z0.size(-1) == self.dof_ndim
        assert z0.size(-2) == self.n_dof
        bs = z0.size(0)
        #z0 = z0.reshape(N, -1)  # -> N x (2 * n_dof * dof_ndim) =: N x D
    
        xp0 = self.to_pos_momentum(z0, models=models)

        self.nfe = 0
        
        if models is None:
            _dynamics = self.forward
        else:
            _dynamics = lambda t, z: torch.stack([model(t, z) for model in models], 0).mean(0)

        xpt = odeint(_dynamics, xp0, ts, rtol=tol, method=method)

        xpt = xpt.permute(1, 0, 2)  # T x bs x D -> bs x T x D
        xpt = xpt.reshape(bs, len(ts), 2, self.n_dof, self.dof_ndim)
        xt, pt = xpt.chunk(2, dim=-3)
        # TODO: make Minv @ pt faster by L(L^T @ pt)
        
        if models is None:
            vt = self.Minv(pt)  # Minv [n_dof x n_dof]. pt [bs, T, 1, n_dof, dof_ndim]
        else:
            vt = torch.stack([model.Minv(pt) for model in models], 0).mean(0)

        xvt = torch.cat([xt, vt], dim=-3)

        if w_div:
            return xvt, delta_logp
        else:
            return xvt

    def collect_model(self):
        self.swag_d_moments.collect_model(self.d_moments)

    def sample(self):
        self.swag_d_moments.sample()

class CHNN(CH):
    def __init__(self,G,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        hidden_size: int = 256,
        num_layers=3,
        wgrad=True,
        **kwargs
    ):
        super().__init__(G=G, dof_ndim=dof_ndim, angular_dims=angular_dims, wgrad=wgrad, **kwargs
        )
        n = len(G.nodes())
        chs = [n * self.dof_ndim] + num_layers * [hidden_size]
        
        layers = \
            [FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)] + \
            [Linear(chs[-1], 1, zero_bias=False, orthogonal_init=True)] + \
            [Reshape(-1)]

        self.potential_net = nn.Sequential(*layers)

        wrap = lambda: nn.Sequential(*layers)
        self.swag_potential_net = SWAG(wrap)

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
        Args:
            x: (N x n_dof x dof_ndim) sized Tensor representing the position in
            Cartesian coordinates
        Returns: a length N Tensor representing the potential energy
        """
        assert x.ndim == 3
        return self.potential_net(x.reshape(x.size(0), -1))

    def compute_V_swag(self, x):
        assert x.ndim == 3
        return self.swag_potential_net(x.reshape(x.size(0), -1))

    def collect_model(self):
        super().collect_model()
        self.swag_potential_net.collect_model(self.potential_net)

    def sample(self):
        super().collect_model()
        self.swag_potential_net.sample()

class AleatoricCHNN(CH):
    def __init__(self,G,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        hidden_size: int = 256,
        num_layers=3,
        wgrad=True,
        **kwargs
    ):
        super().__init__(G=G, dof_ndim=dof_ndim, angular_dims=angular_dims, wgrad=wgrad, **kwargs
        )
        n = len(G.nodes())
        self.diag_cov_dim = 2 * n * dof_ndim

        chs = [n * self.dof_ndim] + num_layers * [hidden_size]
        layers = \
            [FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True)
                for i in range(num_layers)] + \
            [Linear(chs[-1], 1 + self.diag_cov_dim, zero_bias=False, orthogonal_init=True)] 

        self.output_net = nn.Sequential(*layers)

        print(self.output_net)

        chs = [self.diag_cov_dim] + [hidden_size]
        layers = \
            [FCtanh(chs[0], chs[1], zero_bias=False, orthogonal_init=True)] + \
            [Linear(chs[1], self.diag_cov_dim, zero_bias=False, orthogonal_init=True)] 

        self.cov_net = nn.Sequential(*layers)

        print(self.cov_net)

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
        Args:
            x: (N x n_dof x dof_ndim) sized Tensor representing the position in
            Cartesian coordinates
        Returns: a length N Tensor representing the potential energy
        """
        assert x.ndim == 3
        out =  self.output_net(x.reshape(x.size(0), -1))
        return out[:,0]

    def _get_cov_v1(self, zs, ts):
        #batched = zt_pred[:,:,0,:,:].reshape(-1, *zt_pred.shape[3:])
        out =  self.output_net(x.reshape(x.size(0), -1))
        return out[:,1:].exp()

    def _get_cov_v2(self, zs, ts):
        batched = zs.reshape(-1, *zs.shape[2:])
        flat = batched.reshape(batched.size(0), -1)
        ts_rep = ts.expand(zs.size(0),-1).reshape(-1, 1)
        w_t = flat #torch.cat([flat, ts_rep], dim=1)
        cov = self.cov_net(w_t).exp()
        return cov

    def get_covariance(self, zs, ts):
        return self._get_cov_v2(zs, ts)

    def _nll_v1(self, true_zs, z0, ts, tol):
        pred_zs = self.integrate(z0, ts, tol=tol)

        batched = true_zs[:,:,0,:,:].reshape(-1, *true_zs.shape[3:])
        covariance = self.get_covariance(batched)

        mu = pred_zs.reshape(pred_zs.size(0)*pred_zs.size(1), -1)
        dist = torch.distributions.MultivariateNormal(mu, covariance.diag_embed())
        target = true_zs.reshape(true_zs.size(0)*true_zs.size(1), -1)
        nll = -1 * dist.log_prob(target).mean()

        return nll

    def _nll_v2(self, true_zs, z0, ts, tol):
        pred_zs = self.integrate(z0, ts, tol=tol)
        covariance = self.get_covariance(pred_zs.detach(), ts)

        mu = pred_zs.reshape(pred_zs.size(0)*pred_zs.size(1), -1)
        dist = torch.distributions.MultivariateNormal(mu, covariance.diag_embed())
        target = true_zs.reshape(true_zs.size(0)*true_zs.size(1), -1)
        nll = -1 * dist.log_prob(target).mean()

        return nll

    def nll(self, true_zs, z0, ts, tol):
        return self._nll_v2(true_zs, z0, ts, tol)
