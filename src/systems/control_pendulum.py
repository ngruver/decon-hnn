import torch
import torch.nn as nn
import numpy as np
from oil.utils.utils import export

from src.systems.chain_pendulum import ChainPendulum
from src.dynamics import ConstrainedHamiltonianDynamics

class AlternatingPolicy(nn.Module):

    def __init__(
        self
    ):
        super().__init__()
        self.ctr = 0

    def forward(self, t, z):
        num_buckets = 8
        action = abs(hash(str(t) + str(z))) % num_buckets
        
        if action == 0:
            action = -1
        elif action == num_buckets - 1:
            action = 1
        else:
            action = 0

        action = torch.from_numpy(np.array([action])).to(z)
        action = action.unsqueeze(0).expand(z.shape[0], -1)
        return action

class ControlDynamics(nn.Module):

    def __init__(
        self,
        control_interval=0.3,
    ):
        super().__init__()
        self.policy = AlternatingPolicy()

    def forward(self, t, z):
        bs, D = z.shape

        action = self.policy(t, z)

        control_dynamics = torch.zeros_like(z)
        control_dynamics[:, (D // 2):(D // 2) + 1] = 100 * action

        return control_dynamics

@export
class ControlChainPendulum(ChainPendulum):

    def __init__(self, links=2, beams=False, m=None, l=None, control_interval=0.3):
        super().__init__(links, beams, m, l)

        self.c_dynamics = ControlDynamics(control_interval)

    def dynamics(self, wgrad=False):
        return ConstrainedHamiltonianDynamics(self.hamiltonian, self.DPhi, wgrad=wgrad,
                                              extra_dynamics=lambda t, z: self.c_dynamics.forward(t,z))