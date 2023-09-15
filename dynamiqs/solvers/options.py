from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .._utils import to_device
from ..utils.tensor_types import dtype_complex_to_real, get_cdtype

__all__ = [
    'Propagator',
    'Dopri5',
    'Euler',
    'Rouchon1',
    'Rouchon2',
]


class Options:
    GRADIENT_ALG = []

    def __init__(
        self,
        *,
        gradient_alg: str | None = None,
        save_states: bool = True,
        verbose: bool = True,
        dtype: torch.complex64 | torch.complex128 | None = None,
        device: str | torch.device | None = None,
    ):
        """...

        Args:
            gradient_alg (str, optional): Algorithm used for computing gradients.
                Defaults to `None`.
            save_states (bool, optional): If `True`, the state is saved at every
                time value. If `False`, only the final state is stored and returned.
                Defaults to `True`.
            dtype (torch.dtype, optional): Complex data type to which all complex-valued
                tensors are converted. `t_save` is also converted to a real data type of
                the corresponding precision.
            device (string or torch.device, optional): Device on which the tensors are
                stored.
        """
        self.gradient_alg = gradient_alg
        self.save_states = save_states
        self.verbose = verbose
        self.cdtype = get_cdtype(dtype)
        self.rdtype = dtype_complex_to_real(self.cdtype)
        self.device = to_device(device)

        # check that the gradient algorithm is supported
        if self.gradient_alg is not None and self.gradient_alg not in self.GRADIENT_ALG:
            available_gradient_alg_str = ', '.join(f'"{x}"' for x in self.GRADIENT_ALG)
            raise ValueError(
                f'Gradient algorithm "{self.gradient_alg}" is not supported by solver'
                f' `{type(self).__name__}` (supported: {available_gradient_alg_str}).'
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            'gradient_alg': self.gradient_alg,
            'save_states': self.save_states,
            'verbose': self.verbose,
            'dtype': self.cdtype,
            'device': self.device,
        }

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        attributes_str = ', '.join(f'{k}={v}' for k, v in self.as_dict().items())
        return f'{type(self).__name__}({attributes_str})'


class AutogradOptions(Options):
    GRADIENT_ALG = ['autograd']


class AdjointOptions(AutogradOptions):
    GRADIENT_ALG = ['autograd', 'adjoint']

    def __init__(self, *, parameters: tuple[nn.Parameter, ...] | None = None, **kwargs):
        """

        Args:
            parameters (tuple of nn.Parameter): Parameters with respect to which
                gradients are computed during the adjoint state backward pass.
        """
        super().__init__(**kwargs)
        self.parameters = parameters

        # check parameters were passed if gradient by the adjoint
        if self.gradient_alg == 'adjoint' and self.parameters is None:
            raise ValueError(
                'Missing argument `parameters` for gradient algorithm "adjoint".'
            )

    def as_dict(self) -> dict[str, Any]:
        return super().as_dict() | {'parameters': self.parameters}


class ODEFixedStep(AutogradOptions):
    def __init__(self, *, dt: float, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt

    def as_dict(self) -> dict[str, Any]:
        return super().as_dict() | {'dt': self.dt}


class ODEAdaptiveStep(AutogradOptions):
    def __init__(
        self,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_steps: int = 100_000,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor

    def as_dict(self) -> dict[str, Any]:
        return super().as_dict() | {
            'atol': self.atol,
            'rtol': self.rtol,
            'max_steps': self.max_steps,
            'safety_factor': self.safety_factor,
            'min_factor': self.min_factor,
            'max_factor': self.max_factor,
        }


class Propagator(AutogradOptions):
    pass


class Dopri5(ODEAdaptiveStep):
    pass


class Euler(ODEFixedStep):
    pass


class Rouchon1(ODEFixedStep, AdjointOptions):
    def __init__(self, *, sqrt_normalization: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.sqrt_normalization = sqrt_normalization


class Rouchon2(ODEFixedStep, AdjointOptions):
    pass


class Rouchon1SME(ODEFixedStep):
    pass