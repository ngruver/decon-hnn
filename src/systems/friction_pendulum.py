import torch
import torch.nn as nn
from oil.utils.utils import export

from src.systems.chain_pendulum import ChainPendulum
from src.systems.spring_pendulum import SpringPendulum
from src.dynamics import ConstrainedHamiltonianDynamics

class FrictionDynamics(nn.Module):

    def __init__(
        self,
        f_const,
    ):
        super().__init__()
        self.f_const = f_const

    def forward(self, t, z):
        bs, D = z.shape

        friction_dynamics = torch.zeros_like(z)
        friction_dynamics[:, D // 2:] = - self.f_const * z[:, D // 2 :]
        
        return friction_dynamics

@export
class FrictionChainPendulum(ChainPendulum):

    def __init__(self, links=2, beams=False, m=None, l=None, f_const=0.5):#0.2):
        super().__init__(links, beams, m, l)

        self.f_const = f_const

    def dynamics(self, wgrad=False):
        f_dynamics = FrictionDynamics(self.f_const)
        return ConstrainedHamiltonianDynamics(self.hamiltonian, self.DPhi, wgrad=wgrad,
                                              extra_dynamics=lambda t, z: f_dynamics.forward(t,z))

@export
class FrictionSpringPendulum(SpringPendulum):

    def __init__(self, bobs=2, m=None, l=1, k=10, f_const=0.5):#0.2):
        super().__init__(bobs, m, l, k)

        self.f_const = f_const

    def dynamics(self, wgrad=False):
        f_dynamics = FrictionDynamics(self.f_const)
        return ConstrainedHamiltonianDynamics(self.hamiltonian, self.DPhi, wgrad=wgrad,
                                              extra_dynamics=lambda t, z: f_dynamics.forward(t,z))