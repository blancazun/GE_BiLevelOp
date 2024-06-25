from dataclasses import replace
import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover

class OpMutation(Mutation):
    def __init__(self, size=3):
        """

        This mutation is applied to permutations. It randomly selects a segment of a chromosome and reverse its order.
        For instance, for the permutation `[1, 2, 3, 4, 5]` the segment can be `[2, 3, 4]` which results in `[1, 4, 3, 2, 5]`.

        """
        super().__init__()
        self.size = size

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i, y in enumerate(X):
            ind = np.random.choice(problem.n_var, size=self.size, replace=False)
            ind2 = np.roll(ind, 1)
            Y[i][ind] = Y[i][ind2]
        return Y


class PMX(Crossover):
    def __init__(self, shift=False, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.shift = shift

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for i in range(n_matings):
            a, b = X[:, i, :]
            pos1, pos2 = np.sort(np.random.choice(n_var, size=2, replace=False))

            ind1 = np.full(n_var, -1)
            ind2 = np.full(n_var, -1)

            ind1[pos1:pos2] = b[pos1:pos2]
            ind2[pos1:pos2] = a[pos1:pos2]

            for from_value, to_value in zip(b[pos1:pos2], a[pos1:pos2]):
                a[a == from_value] = to_value
                
            for from_value, to_value in zip(a[pos1:pos2], b[pos1:pos2]):
                b[b == from_value] = to_value

            ind1[:pos1] = a[:pos1]
            ind2[:pos1] = b[:pos1]
            ind1[pos2:] = a[pos2:]
            ind2[pos2:] = b[pos2:]

            Y[0, i, :] = ind1
            Y[1, i, :] = ind2

        return Y
