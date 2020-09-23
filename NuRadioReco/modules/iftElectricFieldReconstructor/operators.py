import numpy as np
import nifty5 as ift


class LinearSlopeOperator(ift.LinearOperator):
    def __init__(self, target):
        self._target = ift.DomainTuple.make(target)
        self._domain = ift.DomainTuple.make(ift.UnstructuredDomain((2,)))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        pos = self.target[0].get_k_length_array().val

        self._pos = pos

    def apply(self, x, mode):
        self._check_input(x, mode)
        inp = x.to_global_data()
        if mode == self.TIMES:
            res = np.empty(self.target.shape, dtype=x.dtype)
            res = inp[1] + inp[0] * self._pos
        else:
            res = np.array(
                [np.sum(self._pos * inp),
                 np.sum(inp[1:])], dtype=x.dtype)
        return ift.Field.from_global_data(self._tgt(mode), res)


def SlopeSpectrumOperator(target, m=0, n=0, sigma_m=.1, sigma_n=.1):
    codomain = target.get_default_codomain()

    pos_diagonals = np.ones(target.shape[0])
    pos_diagonals[target.shape[0] // 2 + 1:] = -1
    flipper = ift.DiagonalOperator(ift.Field(ift.DomainTuple.make(codomain), pos_diagonals))
    slope = LinearSlopeOperator(target.get_default_codomain())
    mean = np.array([m, n])
    sig = np.array([sigma_m, sigma_n])
    mean = ift.Field.from_global_data(slope.domain, mean)
    sig = ift.Field.from_global_data(slope.domain, sig)
    linear_operator = flipper @ slope @ ift.Adder(mean) @ ift.makeOp(sig)
    return linear_operator.ducktape('slope')


class Inserter(ift.LinearOperator):
    def __init__(self, target):
        self._domain = ift.makeDomain(ift.UnstructuredDomain(1))
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            return ift.full(self.target, x[0])
        return ift.full(self.domain, x.sum())


class DomainFlipper(ift.LinearOperator):
    """
    Operator that changes a field's domain to its default codomain
    """
    def __init__(self, domain, target=None):
        self._domain = ift.DomainTuple.make(domain)
        if target is None:
            self._target = ift.DomainTuple.make(domain.get_default_codomain())
        else:
            self._target = ift.DomainTuple.make(target)
        self._capability = self._all_ops
        return

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            y = ift.from_global_data(self._target, x.to_global_data())
        if mode == self.INVERSE_TIMES:
            y = ift.from_global_data(self._domain, x.to_global_data())
        if mode == self.ADJOINT_TIMES:
            y = ift.from_global_data(self._domain, x.to_global_data())
        if mode == self.ADJOINT_INVERSE_TIMES:
            y = ift.from_global_data(self._target, x.to_global_data())
        return y


class SymmetrizingOperator(ift.EndomorphicOperator):
    """Adds the field axes-wise in reverse order to itself.

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        Domain of the operator.
    space : int
        Index of space in domain on which the operator shall act. Default is 0.
    """
    def __init__(self, domain, space=0):
        self._domain = ift.DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._space = ift.utilities.infer_space(self._domain, space)

    def apply(self, x, mode):
        self._check_input(x, mode)
        v = x.val.copy()
        for i in self._domain.axes[self._space]:
            lead = (slice(None),) * i
            v, loc = ift.dobj.ensure_not_distributed(v, (i,))
            loc[lead + (slice(None),)] += loc[lead + (slice(None, None, -1),)]
            loc /= 2
        return ift.Field(self.target, ift.dobj.ensure_default_distributed(v))
