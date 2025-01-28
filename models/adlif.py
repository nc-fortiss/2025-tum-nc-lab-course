from typing import Tuple
import torch
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
#from models.helpers import SLAYER
#from module.tau_trainers import TauTrainer, get_tau_trainer_class
#from omegaconf import DictConfig

# from Advancing Spatio-Temporal Processing in Spiking Neural Networks through Adaptation
# https://arxiv.org/pdf/2408.07517
# code: https://github.com/IGITUGraz/SE-adlif/blob/main/models/alif.py

def get_tau_trainer_class(name: str):
    if name == "interpolation":
        return InterpolationTrainer
    elif name == "fixed":
        return FixedTau
    else:
        raise ValueError("Invalid tau trainer name: " + name)


class TauTrainer(Module):
    __constants__ = ["in_features"]
    weight: torch.Tensor
    def __init__(
        self,
        in_features: int,        
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs
        ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super(TauTrainer, self).__init__(**kwargs)
        self.dt = dt        
        self.weight = Parameter(torch.empty(in_features, **factory_kwargs))
        self.register_buffer("tau_max", torch.tensor(tau_max, **factory_kwargs))
        self.register_buffer("tau_min", torch.tensor(tau_min, **factory_kwargs))
        
        
        
    def reset_parameters(self) -> None:
        raise NotImplementedError("This function should not be call from the base class.")
    
    def apply_parameter_constraints(self) -> None:
        raise NotImplementedError("This function should not be call from the base class.")
    
    def forward(self) -> torch.Tensor:
        raise  NotImplementedError("This function should not be call from the base class.")
    
    def get_tau(self) -> torch.Tensor:
        raise  NotImplementedError("This function should not be call from the base class.")
    
    def get_decay(self):
        return self.forward()

class FixedTau(TauTrainer):
    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super(FixedTau, self).__init__(
            in_features, dt, tau_min, tau_max, device, dtype, **kwargs)
        
    def apply_parameter_constraints(self):
        pass

    def forward(self):
        return torch.exp(-self.dt / self.get_tau())

    def get_tau(self):
        return self.weight

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, a=self.tau_min, b=self.tau_max)
        self.weight.requires_grad = False
        


class InterpolationTrainer(TauTrainer):
    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(in_features, dt, tau_min, tau_max, device, dtype, **kwargs)
    def apply_parameter_constraints(self):
        with torch.no_grad():
            self.weight.clamp_(0.0, 1.0)

    def forward(self):
        return torch.exp(-self.dt / self.get_tau())

    def get_tau(self):
        return  self.weight * self.tau_max + (1.0 - self.weight) * self.tau_min

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)
        self.weight.requires_grad = True

# SLAYER surrogate gradient function
def SLAYER(x, alpha, c):
    return c * alpha / (2 * torch.exp(x.abs() * alpha))

class LI_Neuron(Module):
    __constants__ = ["features"]
    features: int

    def __init__(
        self,
        cfg,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.features = cfg['n_neurons']
        self.dt = 1.0
        self.tau_u_range = cfg['tau_u_range']
        self.train_tau_u_method = 'interpolation'

        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
            self.features,
            self.dt,
            self.tau_u_range[0],
            self.tau_u_range[1],
            **factory_kwargs,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.tau_u_trainer.reset_parameters()

    @torch.jit.ignore
    def extra_repr(self) -> str:
        return "features={}".format(
            self.features is not None
        )

    def initial_state(self, x, device) -> Tensor:
        #size = (batch_size, self.features)
        #u = torch.zeros(size=size, device=device, dtype=torch.float, requires_grad=True)
        u = torch.zeros_like(x, device=device, dtype=torch.float, requires_grad=True)
        return u

    def forward(self, input_tensor: Tensor, states: Tensor) -> Tuple[Tensor, Tensor]:
        u_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        current = input_tensor

        if len(current.shape) == 4:
            soma_current = current.permute(0, 2, 3, 1)

        if len(u_tm1.shape) == 4:
            u_tm1 = u_tm1.permute(0, 2, 3, 1)

        u_t = decay_u * u_tm1 + (1.0 - decay_u) * current

        if len(u_t.shape) == 4:
            u_t = u_t.permute(0, 3, 1, 2)

        return u_t.clone(), u_t

    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()

class EFAdLIF(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    a: Tensor
    b: Tensor 
    weight: Tensor

    def __init__(
        self,
        cfg, #: Dict,
        device=None,
        dtype=None, #torch.float32,
        **kwargs,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__(**kwargs)
        self.in_features = cfg['input_size']
        self.out_features = cfg['n_neurons']
        self.dt = 1.0
        self.thr = cfg.get('thr', 1.0)
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)
        self.tau_u_range = cfg['tau_u_range']
        self.train_tau_u_method = 'interpolation'
        self.tau_w_range = cfg['tau_w_range']
        self.train_tau_w_method = 'interpolation'        
        self.use_recurrent = cfg.get('use_recurrent', True)
        self.with_linear = cfg.get('with_linear', True)

        self.a_range = [0.0, 1.0]
        self.b_range = [0.0, 2.0]
        
        self.q = cfg['q']
        
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
                self.out_features,
                self.dt, 
                self.tau_u_range[0], 
                self.tau_u_range[1],
                **factory_kwargs)
        
        self.tau_w_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_w_method)(
                self.out_features,
                self.dt, 
                self.tau_w_range[0], 
                self.tau_w_range[1],
                **factory_kwargs)
        
        if self.with_linear:
            self.weight = Parameter(
                torch.empty((self.out_features, self.in_features), **factory_kwargs)
            )
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        
        if self.use_recurrent:
            self.recurrent = Parameter(
                torch.empty((self.out_features, self.out_features), **factory_kwargs)
            )
        else:
            self.register_buffer("recurrent", None)

        self.a = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.b = Parameter(torch.empty(self.out_features, **factory_kwargs))


        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        self.tau_u_trainer.reset_parameters()
        self.tau_w_trainer.reset_parameters()
        
        if self.with_linear:
            torch.nn.init.uniform_(
                self.weight,
                -1.0 * torch.sqrt(1 / torch.tensor(self.in_features)),
                torch.sqrt(1 / torch.tensor(self.in_features)),
            )
            
            torch.nn.init.zeros_(self.bias)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(
                self.recurrent,
                gain=1.0,
            )
        
        torch.nn.init.uniform_(self.a, self.a_range[0], self.a_range[1])
        torch.nn.init.uniform_(self.b, self.b_range[0], self.b_range[1])
        
    def initial_state(self, x, device) -> Tensor:
        #size = (batch_size, self.out_features)
        u = torch.zeros_like(
            x,
            device=device, 
            dtype=torch.float, 
            requires_grad=True
        )
        z = torch.zeros_like(
            x,
            device=device, 
            dtype=torch.float, 
            requires_grad=True
        )
        w = torch.zeros_like(
            x,
            device=device,
            dtype=torch.float,
            requires_grad=True,
        )
        return u, z, w

    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_w_trainer.apply_parameter_constraints()
        self.a.data = torch.clamp(self.a, min=self.a_range[0], max=self.a_range[1])
        self.b.data = torch.clamp(self.b, min=self.b_range[0], max=self.b_range[1])

    def forward(
        self, input_tensor: Tensor,  states: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        u_tm1, z_tm1, w_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        if self.with_linear:
            soma_current = F.linear(input_tensor, self.weight, self.bias)
        else:
            soma_current = input_tensor
        if self.use_recurrent:
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current

        if len(soma_current.shape) == 4:
            soma_current = soma_current.permute(0, 2, 3, 1)

        if len(u_tm1.shape) == 4:
            u_tm1 = u_tm1.permute(0, 2, 3, 1)
            z_tm1 = z_tm1.permute(0, 2, 3, 1)
            w_tm1 = w_tm1.permute(0, 2, 3, 1)

        u_t = decay_u * u_tm1 + (1.0 - decay_u) * (
            soma_current - w_tm1
        )
        
        u_thr = u_t - self.thr
        # Forward Gradient Injection trick (credits to Sebastian Otte)
        z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
        u_t = u_t * (1 - z_t.detach())
        w_t = (
            decay_w * w_tm1
            + (1.0 - decay_w) * (self.a * u_tm1 + self.b * z_tm1) * self.q
        )

        if len(u_t.shape) == 4:
            u_t = u_t.permute(0, 3, 1, 2)
            z_t = z_t.permute(0, 3, 1, 2)
            w_t = w_t.permute(0, 3, 1, 2)

        return z_t.clone(), (u_t, z_t, w_t)
    
class SEAdLIF(EFAdLIF):
    def forward(
        self, input_tensor: Tensor, states: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        u_tm1, z_tm1, w_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        if self.with_linear:
            soma_current = F.linear(input_tensor, self.weight, self.bias)
        else:
            soma_current = input_tensor
        if self.use_recurrent:
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current

        if len(soma_current.shape) == 4:
            soma_current = soma_current.permute(0, 2, 3, 1)

        if len(u_tm1.shape) == 4:
            u_tm1 = u_tm1.permute(0, 2, 3, 1)
            z_tm1 = z_tm1.permute(0, 2, 3, 1)
            w_tm1 = w_tm1.permute(0, 2, 3, 1)
            
        u_t = decay_u * u_tm1 + (1.0 - decay_u) * (
            soma_current - w_tm1
        )
        u_thr = u_t - self.thr
        # Forward Gradient Injection trick (credits to Sebastian Otte)
        z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
        
        # Symplectic formulation with early reset

        u_t = u_t * (1 - z_t.detach())
        w_t = (
            decay_w * w_tm1
            + (1.0 - decay_w) * (self.a * u_t + self.b * z_t) * self.q
        )

        if len(u_t.shape) == 4:
            u_t = u_t.permute(0, 3, 1, 2)
            z_t = z_t.permute(0, 3, 1, 2)
            w_t = w_t.permute(0, 3, 1, 2)

        return z_t.clone(), (u_t, z_t, w_t)
    
class AdLIF(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float,
        rnn=False,
        dtype=None,
        device=torch.device('cpu'),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = {'input_size': in_features,
                    'n_neurons': out_features,
                    'thr': threshold,
                    'alpha': 5.0,
                    'c': 0.4,
                    'tau_u_range': [5, 25],
                    'tau_w_range': [60, 300],
                    'use_recurrent': rnn,
                    'with_linear': False,
                    'q': 120}
        self.neuron = SEAdLIF(self.cfg, device, dtype)
        

    def forward(self, x):
        # multi-step forward for adLIF neuron
        self.neuron.apply_parameter_constraints()
        u, z, w = self.neuron.initial_state(x[0], x.device)
        s = (u, z, w)
        outs = []
        for t in range(x.shape[0]):
            out, s = self.neuron(x[t], s)
            outs.append(out)

        out = torch.stack(outs, dim=0)
        return out


class LI(Module):
    def __init__(
            self,
            out_features: int,
            dtype=None,
            device=torch.device('cpu'),
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = {'n_neurons': out_features,
                    'tau_u_range': [5, 25], }
        self.neuron = LI_Neuron(self.cfg, device, dtype)

    def forward(self, x):
        # multi-step forward for LI neuron
        self.neuron.apply_parameter_constraints()
        u = self.neuron.initial_state(x[0], x.device)
        s = u
        outs = []
        for t in range(x.shape[0]):
            out, s = self.neuron(x[t], s)
            outs.append(out)

        out = torch.stack(outs, dim=0)
        return out