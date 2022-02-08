import math
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import numpy as np

class Swish(nn.Module):
    def forward(self,x):
        return x*x.sigmoid()

def FCtanh(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(Linear(chin, chout, zero_bias, orthogonal_init), nn.Tanh())


def FCswish(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(Linear(chin, chout, zero_bias, orthogonal_init), Swish())


def FCsoftplus(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(Linear(chin, chout, zero_bias, orthogonal_init), nn.Softplus())


def Linear(chin, chout, zero_bias=False, orthogonal_init=False):
    linear = nn.Linear(chin, chout)
    if zero_bias:
        torch.nn.init.zeros_(linear.bias)
    if orthogonal_init:
        torch.nn.init.orthogonal_(linear.weight, gain=0.5)
    return linear


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CosSin(nn.Module):
    def __init__(self, q_ndim, angular_dims, only_q=True):
        super().__init__()
        self.q_ndim = q_ndim
        self.angular_dims = tuple(angular_dims)
        self.non_angular_dims = tuple(set(range(q_ndim)) - set(angular_dims))
        self.only_q = only_q

    def forward(self, q_or_qother):
        if self.only_q:
            q = q_or_qother
        else:
            split_dim = q_or_qother.size(-1)
            splits = [self.q_ndim, split_dim - self.q_ndim]
            q, other = q_or_qother.split(splits, dim=-1)
        
        # print("\n")
        # print(q_or_qother.size())
        # print(q.size())
        # print(other.size())
        # print(self.q_ndim)
        # print("")
        assert q.size(-1) == self.q_ndim

        q_angular = q[..., self.angular_dims]
        q_not_angular = q[..., self.non_angular_dims]

        cos_ang_q, sin_ang_q = torch.cos(q_angular), torch.sin(q_angular)
        q = torch.cat([cos_ang_q, sin_ang_q, q_not_angular], dim=-1)

        if self.only_q:
            q_or_other = q
        else:
            q_or_other = torch.cat([q, other], dim=-1)

        return q_or_other


def tril_mask(square_mat):
    n = square_mat.size(-1)
    coords = torch.arange(n)
    return coords <= coords.view(n, 1)


def mod_angles(q, angular_dims):
    assert q.ndim == 2
    D = q.size(-1)
    non_angular_dims = list(set(range(D)) - set(angular_dims))
    # Map to -pi, pi
    q_modded_dims = torch.fmod(q[..., angular_dims] + math.pi, 2 * math.pi) + (2. * (q[..., angular_dims] < -math.pi) - 1) * math.pi
    if (q_modded_dims.abs() > math.pi).any():
        raise RuntimeError("Angles beyond [-pi, pi]!")
    q_non_modded_dims = q[..., non_angular_dims]
    return torch.cat([q_modded_dims, q_non_modded_dims], dim=-1)



# See https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm
class SNLinear(nn.Linear):
    """ Linear layer with spectral normalization and an additional _s parameter for the spectral norm.
        Can add regularization on the _s parameter to implement the SN RPP as discussed."""
    eps=1e-12
    
    def __init__(self,*args,n_power_iterations=1,**kwargs):
        super().__init__(*args,**kwargs)
        self.__s = nn.Parameter(torch.tensor(np.log(np.exp(1)-1))) # parameter to hold the singular value of the layer
        self._u = normalize(self.weight.new_empty(self.out_features).normal_(0, 1), dim=0, eps=self.eps)
        self._v = normalize(self.weight.new_empty(self.in_features).normal_(0, 1), dim=0, eps=self.eps)
        self.n_power_iterations=n_power_iterations
    
    @property
    def _s(self):
        return F.softplus(self.__s)

    def compute_weight(self, do_power_iteration=True) -> torch.Tensor:
        weight_mat = self.weight
        u=self._u
        v=self._v
        if do_power_iteration:
            with torch.no_grad():
                # updates are performed in place 
                # (see https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm)
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), 
                                  u.to(weight_mat)), dim=0, eps=self.eps, out=v.to(weight_mat))
                    u = normalize(torch.mv(weight_mat, 
                                  v.to(weight_mat)), dim=0, eps=self.eps, out=u.to(weight_mat))
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        return self.weight/sigma
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.compute_weight() * self._s
        return F.linear(input, w, self.bias)


def convert_linear_to_snlinear(model):
    """ Converts all linear layers in model to spectral norm versions """
    for child_name, child in model.named_children():
        if type(child)==nn.Linear:
            setattr(model, child_name, SNLinear(child.in_features,child.out_features,bias=(child.bias is not None)))
        else:
            convert_linear_to_snlinear(child)