from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling, PermutationRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.core.problem import Problem
from SR import SR
from GE import Grammar, mapping_depth_first, mapping_breadth_first, mapping_pigrammatical
from os.path import join
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

def evaluate_sr(grammar, sr, x, order=None):
    expr = grammar.mapping(x, order)
    return sr.evaluate(expr)


class UpperLevelProblem(Problem):
    def __init__(self, n_var=-1, xl=None, xu=None, grammar=None, instance=None, **kwargs):
        super().__init__(n_var, n_obj=1, n_constr=0, xl=xl, xu=xu,  **kwargs)
        self.grammar = deepcopy(grammar)
        self.sr_problem = deepcopy(instance)

    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate the lower level problem
        F = []
        per = []

        """
        Inicio de lo que se paralelizar, revisar si con GA o un SA
        Hay que probar operadores de Cruza como OrderCrossOver o el PMX y Mutación como InversionMutation, 2 o 3 opt.
        """
        for item in x:
            llp = LowerLevelProblem(item, xl=0, xu=self.n_var, grammar=self.grammar, instance=self.sr_problem)

            algoritmo = GA(
                pop_size=10,
                eliminate_duplicates=True,
                sampling=PermutationRandomSampling(),
                crossover=OrderCrossover(),
                mutation=InversionMutation(),
            )
            # ("op", OpMutation, dict(size=3))
            termination = get_termination("n_gen", 100)  # ("n_eval", 25000) # get_termination("n_gen", 1000)

            res = minimize(llp,
                           algoritmo,
                           termination,
                           save_history=True,
                           verbose=False)
            """
            print("-" * 40)
            print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
            """
            F.append(res.F)
            per.append(res.X)
        """
        Final de lo que se paralelizar
        """

        #implementar algo... (no se que) para guardar la mejor F y su respectiva per

        out["F"] = np.asarray(F)
        # print(out["F"])


class LowerLevelProblem(Problem):
    def __init__(self, values, xl=None, xu=None, grammar=None, instance=None, **kwargs):
        super().__init__(len(values), n_obj=1, n_constr=0, xl=xl, xu=xu, **kwargs)
        self.grammar = deepcopy(grammar)
        self.sr_problem = deepcopy(instance)
        self.values = deepcopy(values)

    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate the lower level problem
        out["F"] = np.asarray([evaluate_sr(self.grammar, self.sr_problem, self.values, item) for item in x])


if __name__ == '__main__':
    instance_path = "instances"
    instances = ["F3"]  # "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "Keijzer6"


    """
    Hacer el calculo para trabajar con 250,000 elementos
    """
    mappings = [mapping_depth_first, mapping_breadth_first]  # mapping_breadth_first, mapping_depth_first
    val = []
    for mapping in mappings:
        for instance in instances[:2]:
            sr_train = SR(file=join(instance_path, instance, "train.csv"))
            sr_test = SR(file=join(instance_path, instance, "test.csv"))

            grammarP = Grammar(mapping=mapping, file=join(instance_path, instance, "grammar.bnf"))

            ulp = UpperLevelProblem(n_var=100, xl=0.0, xu=255.0, grammar=grammarP,
                                    instance=sr_train)

            algorithm = GA(
                pop_size=10,
                eliminate_duplicates=True,
                sampling=IntegerRandomSampling(),
                crossover=TwoPointCrossover(),
                mutation=BitflipMutation(),
            )

            termination = get_termination("n_eval", 200)

            res = minimize(ulp,
                           algorithm,
                           termination,
                           save_history=True,
                           verbose=True)

            print("-" * 40)
            print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
            results = grammarP.mapping(res.X)
            print(mapping.__name__, instance, results)
            print(res.F, sr_test.evaluate(results))

            val = [e.opt.get("F")[0][0] for e in res.history]
            plt.plot(np.arange(len(val)), val, label=f'{mapping.__name__}, {instance}')
    plt.legend()
    plt.show()