import torch
import torch.nn as nn
from oil.utils.utils import export

from src.systems.chain_pendulum import ChainPendulum
from src.dynamics import ConstrainedHamiltonianDynamics

class ControlDynamics(nn.Module):

    def __init__(
        self,
        control_interval=10,
    ):
        super().__init__()
        self.control_interval = control_interval
        self.ctr = -1

    def forward(self, t, z):
        self.ctr += 1

        control_dynamics = torch.zeros_like(z)

        if self.ctr == 0 or self.ctr % self.control_interval != 0:
            return control_dynamics

        bs, D = z.shape
        control_dynamics[:, (D // 2):(D // 2) + 1] = 200 * (2 * ((self.ctr // 2) % 2) - 1)
        #100 * (2 * (torch.rand(bs,1) < 0.5).float() - 1)

        # print(control_dynamics)

        return control_dynamics

@export
class ControlChainPendulum(ChainPendulum):

    def __init__(self, links=2, beams=False, m=None, l=None, control_interval=10):
        super().__init__(links, beams, m, l)

        self.c_dynamics = ControlDynamics(control_interval)

    def dynamics(self, wgrad=False):
        return ConstrainedHamiltonianDynamics(self.hamiltonian, self.DPhi, wgrad=wgrad,
                                              extra_dynamics=lambda t, z: self.c_dynamics.forward(t,z))