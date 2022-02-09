import torch
import torch.nn as nn
from torchdiffeq import odeint

from src.models.hnn import HNN
from .utils import FCtanh, Reshape, Linear
from ..dynamics.hamiltonian import HamiltonianDynamics, GeneralizedT


def make_mlp(input_size, hidden_size, output_size, depth):
    if depth > 0:
        chs = [input_size] + depth * [hidden_size]
        net = nn.Sequential(
            *[FCtanh(chs[i], chs[i + 1], zero_bias=False, orthogonal_init=True) for i in range(depth)],
            Linear(chs[-1], output_size),
        )
    else:
        net = Linear(input_size, output_size)
    return net


class RecurrentNetwork(nn.Module):
    def __init__(self, input_size, output_size, enc_hidden_size, rec_hidden_size, dec_hidden_size,
                 enc_depth, rec_depth, dec_depth, **kwargs):
        super().__init__()
        self.encoder = make_mlp(input_size, enc_hidden_size, rec_hidden_size, enc_depth)
        self.recurrent = nn.GRU(rec_hidden_size, rec_hidden_size, num_layers=rec_depth)
        self.decoder = make_mlp(rec_hidden_size, dec_hidden_size, output_size, dec_depth)
        self.rec_hidden_size = rec_hidden_size
        self.rec_depth = rec_depth

        self.register_buffer('input_loc', torch.zeros(input_size))
        self.register_buffer('input_scale', torch.ones(input_size))
        self.register_buffer('output_loc', torch.zeros(output_size))
        self.register_buffer('output_scale', torch.ones(output_size))

        self.nfe = 0
        self._hidden_state = None
        self._weight_decay = kwargs.setdefault('weight_decay', 0.)

    def init_hidden_state(self, inputs):
        assert inputs.dim() == 2
        n_batch = inputs.size(0)
        h = torch.zeros(self.rec_depth, n_batch, self.rec_hidden_size).to(inputs)
        return h
        
    def step(self, inputs, hidden_state):
        assert inputs.dim() == 2
        inputs_emb = self.encoder(inputs).unsqueeze(0)
        outputs_emb, hidden_state = self.recurrent(inputs_emb, hidden_state)
        outputs = self.decoder(outputs_emb.squeeze(0))
        self.nfe += 1
        return outputs, hidden_state

    def forward(self, t, z):
        net_input = (z - self.input_loc) / self.input_scale
        net_output, self._hidden_state = self.step(net_input, self._hidden_state)
        pred_z = self.output_scale * net_output + self.output_loc
        return pred_z

    def integrate(self, x_0, ts, u=None, reset_hidden=True, tol=None):
        self.nfe = 0
        pred_x = [x_0.clone()]
        original_shape = x_0.shape
        x_0 = x_0.view(original_shape[0], -1)
        if reset_hidden or self._hidden_state is None:
            self._hidden_state = self.init_hidden_state(x_0)
        x_t = x_0
        for t_idx, t_val in enumerate(ts[:-1]):
            if u is None:
                z_t = x_t
            else:
                z_t = torch.cat((x_t, u[:, t_idx]), dim=-1)
            x_t = self(t_val, z_t)
            pred_x.append(x_t.view(*original_shape))
        return torch.stack(pred_x, dim=1)

    @property
    def param_groups(self):
        return [{'params': self.parameters(), 'weight_decay': self._weight_decay}]


class NODE(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size=256, num_layers=2, method='euler',
        **kwargs
    ):
        super().__init__()

        chs = [input_size] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], True, True) for i in range(num_layers)]
        activations = [nn.Tanh() for i in range(num_layers)]
        self.net = nn.Sequential(
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], output_size, True, True),
        )
        self.u = None
        self._int_method = method
        self._step_size = kwargs.setdefault('step_size', 1.)
        self._weight_decay = kwargs.setdefault('weight_decay', 0.)

        self.register_buffer('input_loc', torch.zeros(input_size))
        self.register_buffer('input_scale', torch.ones(input_size))
        self.register_buffer('output_loc', torch.zeros(output_size))
        self.register_buffer('output_scale', torch.ones(output_size))

        self.nfe = 0
        
    def forward(self, t, z):
        """assumes x.shape == (batch_size, (q_size + v_size)), with elements [pos, vel]"""
        if self.u is None:
            net_input = z
        else:
            diff = (self.ts - t).pow(2)[self.ts <= t]
            neigh = self.ts[self.ts <= t][diff.argmin()]
            u_t = self.u[neigh.item()]
            net_input = torch.cat([z, u_t], dim=-1)

        net_input = (net_input - self.input_loc) / self.input_scale
        dz_dt = self.output_scale * self.net(net_input) + self.output_loc

        self.nfe += 1
        return dz_dt

    def integrate(self, x_0, ts, u=None, tol=1e-4):
        self.nfe = 0
        if u is None:
            self.u = None
        else:
            self.u = {t.item(): u_t for t, u_t in zip(ts, u.permute(1, 0, 2))}
            self.u[ts[-1].item()] = u[:, -1]

        self.ts = ts
        original_shape = x_0.shape
        x_0 = x_0.view(original_shape[0], -1)
        odeint_options = dict(step_size=self._step_size, interp='linear')
        pred_x = odeint(self.forward, x_0, ts, rtol=tol, method=self._int_method, options=odeint_options)
        pred_x = pred_x.view(-1, *original_shape).transpose(1, 0)  # T x N x D -> N x T x D
        return pred_x
    
    @property
    def param_groups(self):
        return [{'params': self.parameters(), 'weight_decay': self._weight_decay}]
    

class CoupledNODE(NODE):
    def forward(self, t, z):
        z_size = z.size(-1)
        v_t = z[..., (z_size // 2):]
        if self.u is None:
            net_input = z
        else:
            diff = (self.ts - t).pow(2)[self.ts <= t]
            neigh = self.ts[self.ts <= t][diff.argmin()]
            u_t = self.u[neigh.item()]
            net_input = torch.cat([z, u_t], dim=-1)
        net_input = (net_input - self.input_loc) / self.input_scale
        net_output = self.net(net_input)
        dv_dt = self.output_scale * net_output + self.output_loc
        dq_dt = (2 * v_t + self._step_size * dv_dt) / 2  # approx. RK2 step using Euler estimate of v_{t+1}
        dz_dt = torch.cat((dq_dt, dv_dt), dim=-1)
        self.nfe += 1
        return dz_dt


class MechanicsNN(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size=256, num_layers=2, method='euler',
        **kwargs
    ):
        super().__init__()

        chs = [input_size] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], True, True) for i in range(num_layers)]
        activations = [nn.Tanh() for i in range(num_layers)]
        # dropouts = [nn.Dropout(0.1) for i in range(num_layers)]
        self.dynamics_net = nn.Sequential(
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], output_size // 2, True, True),
        )

        self.velocity_head = nn.Linear(output_size // 2, output_size // 2)

        chs = [output_size // 2] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], True, True) for i in range(num_layers)]
        activations = [nn.Tanh() for i in range(num_layers)]
        # dropouts = [nn.Dropout(0.1) for i in range(num_layers)]
        self.mass_net = nn.Sequential(
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], output_size * output_size // 4, True, True),
            Reshape(-1, output_size // 2, output_size // 2)
        )

        self.u = None
        self._int_method = method
        self._step_size = kwargs.setdefault('step_size', 1.)
        self._weight_decay = kwargs.setdefault('weight_decay', 0.)
        
        self.register_buffer('input_loc', torch.zeros(input_size))
        self.register_buffer('input_scale', torch.ones(input_size))
        self.register_buffer('output_loc', torch.zeros(output_size // 2))
        self.register_buffer('output_scale', torch.ones(output_size // 2))

        self.nfe = 0

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

    def Minv(self, q, eps=1e-4, verbose=False):
        """Compute the learned inverse mass matrix M^{-1}(q)
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q) * 0
        assert lower_triangular.ndim == 3
        eps = 1
        diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=q.dtype, device=q.device)
        Minv = lower_triangular.matmul(lower_triangular.transpose(-2, -1)) + diag_noise
        return Minv

    def M(self, q, eps=1e-6, verbose=False):
        """Returns a function that multiplies the mass matrix M(q) by a vector qdot
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q) * 0
        assert lower_triangular.ndim == 3

        def M_func(qdot):
            assert qdot.ndim == 2
            qdot = qdot.unsqueeze(-1)
            eps = 1
            diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=qdot.dtype, device=qdot.device)
            M = lower_triangular @ lower_triangular.transpose(-2, -1) + diag_noise
            M_times_qdot = torch.solve(qdot, M).solution.squeeze(-1)
            return M_times_qdot

        return M_func

    def forward(self, t, z):
        """assumes x.shape == (batch_size, (q_size + v_size)), with elements [pos, vel]"""
        
        if self.u is None:
            net_input = z
        else:
            diff = (self.ts - 1000 * t).pow(2)[self.ts <= 1000 * t]
            neigh = self.ts[self.ts <= 1000 * t][diff.argmin()]
            u_t = self.u[neigh.item()]
            net_input = torch.cat([z, u_t], dim=-1)

        net_input = (net_input - self.input_loc) / self.input_scale

        q, p = z.chunk(2, dim=-1)
        v = self.Minv(q).matmul(p.unsqueeze(-1)).squeeze(-1)
        dq_dt = self.velocity_head(v)
        dp_dt = self.output_scale * self.dynamics_net(net_input) + self.output_loc
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)

        self.nfe += 1
        return dz_dt

    def integrate(self, z0, ts, u=None, tol=1e-4):
        self.nfe = 0

        self.ts = 1000 * ts
        if u is None:
            self.u = None
        else:
            self.u = {t.item(): u_t for t, u_t in zip(self.ts, u.permute(1, 0, 2))}
            self.u[self.ts[-1].item()] = u[:, -1]
            self.u[self.ts[-2].item()] = u[:, -1]

        q0, v0 = z0.chunk(2, dim=-1)

        p0 = self.M(q0)(v0) #(DxD)*(bsxD) -> (bsxD)
        qp0 = torch.cat([q0, p0], dim=-1)

        qpt = odeint(self.forward, qp0, ts, rtol=tol, method=self._int_method)#, options=odeint_options)
        qpt = qpt.permute(1, 0, 2)  # T x N x D -> N x T x D

        qt, pt = qpt.reshape(-1, z0.shape[-1]).chunk(2, dim=-1)
        vt = self.Minv(qt, verbose=True).matmul(pt.unsqueeze(-1)).squeeze(-1)

        qvt = torch.cat([qt, vt], dim=-1).reshape(*qpt.shape)

        return qvt
    
    @property
    def param_groups(self):
        return [{'params': self.parameters(), 'weight_decay': self._weight_decay}]


class MixtureHNN(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size=256, num_layers=2, method='euler',
        **kwargs
    ):
        super().__init__()

        chs = [output_size // 2] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], True, True) for i in range(num_layers)]
        activations = [nn.Tanh() for i in range(num_layers)]
        # dropouts = [nn.Dropout(0.1) for i in range(num_layers)]
        self.potential_net = nn.Sequential(
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], 1, True, True),
            Reshape(-1)
        )

        chs = [output_size // 2] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], True, True) for i in range(num_layers)]
        activations = [nn.Tanh() for i in range(num_layers)]
        # dropouts = [nn.Dropout(0.1) for i in range(num_layers)]
        self.mass_net = nn.Sequential(
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], output_size * output_size // 4, True, True),
            Reshape(-1, output_size // 2, output_size // 2)
        )

        chs = [input_size] + num_layers * [hidden_size]
        linears = [Linear(chs[i], chs[i + 1], True, True) for i in range(num_layers)]
        activations = [nn.Tanh() for i in range(num_layers)]
        # dropouts = [nn.Dropout(0.1) for i in range(num_layers)]
        self.force_net = nn.Sequential(
            *[val for pair in zip(linears, activations) for val in pair],
            Linear(chs[-1], output_size // 2, True, True),
        )

        self.dynamics = HamiltonianDynamics(self.H, wgrad=True)


        self.u = None
        self._int_method = method
        self._step_size = kwargs.setdefault('step_size', 1.)
        self._weight_decay = kwargs.setdefault('weight_decay', 0.)
        
        self.register_buffer('input_loc', torch.zeros(input_size))
        self.register_buffer('input_scale', torch.ones(input_size))
        self.register_buffer('output_loc', torch.zeros(output_size // 2))
        self.register_buffer('output_scale', torch.ones(output_size // 2))

        self.nfe = 0

    def H(self, t, z):
        """ Compute the Hamiltonian H(t, q, p)
        Args:
            t: Scalar Tensor representing time
            z: N x D Tensor of the N different states in D dimensions.
                Assumes that z is [q, p].
        Returns: Size N Hamiltonian Tensor
        """
        assert (t.ndim == 0) and (z.ndim == 2)
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
        res = res.transpose(-1, -2)  # Make lower triangular
        return res

    def Minv(self, q, eps=1e-4, verbose=False):
        """Compute the learned inverse mass matrix M^{-1}(q)
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q) * 0
        assert lower_triangular.ndim == 3
        eps = 1
        diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=q.dtype, device=q.device)
        Minv = lower_triangular.matmul(lower_triangular.transpose(-2, -1)) + diag_noise
        return Minv

    def M(self, q, eps=1e-6, verbose=False):
        """Returns a function that multiplies the mass matrix M(q) by a vector qdot
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q) * 0
        assert lower_triangular.ndim == 3

        def M_func(qdot):
            assert qdot.ndim == 2
            qdot = qdot.unsqueeze(-1)
            eps = 1
            diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=qdot.dtype, device=qdot.device)
            M = lower_triangular @ lower_triangular.transpose(-2, -1) + diag_noise
            M_cholesky = torch.linalg.cholesky(M)
            M_times_qdot = torch.cholesky_solve(qdot, M_cholesky).squeeze(-1)
            # M_times_qdot = torch.solve(qdot, M).solution.squeeze(-1)
            return M_times_qdot

        return M_func

    def forward(self, t, z):
        """assumes x.shape == (batch_size, (q_size + v_size)), with elements [pos, vel]"""
        
        if self.u is None:
            net_input = z
        else:
            diff = (self.ts - 1000 * t).pow(2)[self.ts <= 1000 * t]
            neigh = self.ts[self.ts <= 1000 * t][diff.argmin()]
            u_t = self.u[neigh.item()]
            net_input = torch.cat([z, u_t], dim=-1)

        net_input = (net_input - self.input_loc) / self.input_scale
        dynamics_input = (z - self.input_loc[:z.shape[-1]]) / self.input_scale[:z.shape[-1]]

        dz_dt = self.dynamics(t, dynamics_input)
        dq_dt, dp_dt = dz_dt.chunk(2, dim=-1)
 
        dp_dt = dp_dt + self.force_net(net_input)
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)

        self.nfe += 1
        return dz_dt

    def integrate(self, z0, ts, u=None, tol=1e-4):
        self.nfe = 0

        self.ts = 1000 * ts
        if u is None:
            self.u = None
        else:
            self.u = {t.item(): u_t for t, u_t in zip(self.ts, u.permute(1, 0, 2))}
            self.u[self.ts[-1].item()] = u[:, -1]
            self.u[self.ts[-2].item()] = u[:, -1]

        q0, v0 = z0.chunk(2, dim=-1)

        p0 = self.M(q0)(v0) #(DxD)*(bsxD) -> (bsxD)
        qp0 = torch.cat([q0, p0], dim=-1)

        qpt = odeint(self.forward, qp0, ts, rtol=tol, method=self._int_method)#, options=odeint_options)
        qpt = qpt.permute(1, 0, 2)  # T x N x D -> N x T x D

        qt, pt = qpt.reshape(-1, z0.shape[-1]).chunk(2, dim=-1)
        vt = self.Minv(qt, verbose=True).matmul(pt.unsqueeze(-1)).squeeze(-1)

        qvt = torch.cat([qt, vt], dim=-1).reshape(*qpt.shape)

        return qvt
    
    @property
    def param_groups(self):
        return [{'params': self.parameters(), 'weight_decay': self._weight_decay}]


class RecNODE(RecurrentNetwork):
    def __init__(self, method='euler', step_size=1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._int_method = method
        self._step_size = step_size

    def forward(self, t, z):
        if self.u is None:
            net_input = z
        else:
            diff = (self.ts - t).pow(2)[self.ts <= t]
            neigh = self.ts[self.ts <= t][diff.argmin()]
            u_t = self.u[neigh.item()]
            net_input = torch.cat([z, u_t], dim=-1)
        net_input = (net_input - self.input_loc) / self.input_scale
        net_output, self._hidden_state = self.step(net_input, self._hidden_state)
        dz_dt = self.output_scale * net_output + self.output_loc
        self.nfe += 1
        return dz_dt

    def integrate(self, x_0, ts, u=None, reset_hidden=True, tol=1e-4):
        self.nfe = 0
        if u is None:
            self.u = None
        else:
            self.u = {t.item(): u_t for t, u_t in zip(ts, u.permute(1, 0, 2))}
            self.u[ts[-1].item()] = u[:, -1]
        self.ts = ts

        original_shape = x_0.shape
        x_0 = x_0.view(original_shape[0], -1)
        if reset_hidden or self._hidden_state is None:
            self._hidden_state = self.init_hidden_state(x_0)
        odeint_options = dict(step_size=self._step_size, interp='linear')
        pred_x = odeint(self.forward, x_0, ts, rtol=tol, method=self._int_method, options=odeint_options)
        pred_x = pred_x.view(-1, *original_shape).transpose(1, 0)  # T x N x D -> N x T x D
        return pred_x


class CoupledRecNODE(RecNODE):
    def forward(self, t, z):
        z_size = z.size(-1)
        v_t = z[..., (z_size // 2):]
        if self.u is None:
            net_input = z
        else:
            diff = (self.ts - t).pow(2)[self.ts <= t]
            neigh = self.ts[self.ts <= t][diff.argmin()]
            u_t = self.u[neigh.item()]
            net_input = torch.cat([z, u_t], dim=-1)
        net_input = (net_input - self.input_loc) / self.input_scale
        net_output, self._hidden_state = self.step(net_input, self._hidden_state)
        dv_dt = self.output_scale * net_output + self.output_loc
        dq_dt = (2 * v_t + self._step_size * dv_dt) / 2  # approx. RK2 step using Euler estimate of v_{t+1}
        dz_dt = torch.cat((dq_dt, dv_dt), dim=-1)
        self.nfe += 1
        return dz_dt


class RPPNet(RecNODE):
    def __init__(self, method, step_size, ham_cfg, *args, **kwargs):
        super().__init__(method, step_size, *args, **kwargs)
        self.ham_net = HNN(**ham_cfg, **kwargs)

        for m in self.decoder.modules():
            if hasattr(m, 'weight') and m.weight.size(0) == kwargs['output_size']:
                torch.nn.init.normal_(m.weight, std=1e-4)
                torch.nn.init.constant_(m.bias, 0.)

    def forward(self, t, x):
        if self.u is None:
            z_t = x
        else:
            diff = (self.ts - t).pow(2)[self.ts <= t]
            neigh = self.ts[self.ts <= t][diff.argmin()]
            u_t = self.u[neigh.item()]
            z_t = torch.cat([x, u_t], dim=-1)
        dx_dt, self._hidden_state = self.step(z_t, self._hidden_state)
        dx_dt += self.ham_net(t, z_t)
        self.nfe += (1 + self.ham_net.nfe)
        return dx_dt

    def integrate(self, x_0, ts, u=None, **kwargs):
        self.ham_net.nfe = 0
        return super().integrate(x_0, ts, u=None, **kwargs)

    @property
    def param_groups(self):
        ham_params, rec_node_params = [], []
        for name, p in self.named_parameters():
            if 'ham_net' in name:
                ham_params.append(p)
            else:
                rec_node_params.append(p)
        groups = [
            {'params': rec_node_params, 'weight_decay': 1e-2},
            {'params': ham_params, 'weight_decay': 0.},
        ]
        return groups
