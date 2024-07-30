import diffrax as dx
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from ..core.abstract_integrator import MCSolveIntegrator
from ..core.diffrax_integrator import (
    DiffraxIntegrator,
    Dopri5Integrator,
    Dopri8Integrator,
    EulerIntegrator,
    Kvaerno3Integrator,
    Kvaerno5Integrator,
    Tsit5Integrator,
)
from ...utils.utils import dag


class MCSolveDiffraxIntegrator(DiffraxIntegrator, MCSolveIntegrator):

    @property
    def terms(self) -> dx.AbstractTerm:
        def vector_field(t: Scalar, state: PyTree, _args: PyTree) -> PyTree:
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(axis=0)
            new_state = -1j * (self.H(t) - 1j * 0.5 * LdL) @ state
            return new_state
        return dx.ODETerm(vector_field)

    @property
    def discrete_terminating_event(self):
        def norm_below_rand(state, **kwargs):
            psi = state.y
            prob = jnp.abs(jnp.einsum("id,id->", jnp.conj(psi), psi))
            return prob < self.rand
        return dx.DiscreteTerminatingEvent(norm_below_rand)


class MCSolveEulerIntegrator(MCSolveDiffraxIntegrator, EulerIntegrator):
    pass


class MCSolveDopri5Integrator(MCSolveDiffraxIntegrator, Dopri5Integrator):
    pass


class MCSolveDopri8Integrator(MCSolveDiffraxIntegrator, Dopri8Integrator):
    pass


class MCSolveTsit5Integrator(MCSolveDiffraxIntegrator, Tsit5Integrator):
    pass


class MCSolveKvaerno3Integrator(MCSolveDiffraxIntegrator, Kvaerno3Integrator):
    pass


class MCSolveKvaerno5Integrator(MCSolveDiffraxIntegrator, Kvaerno5Integrator):
    pass
