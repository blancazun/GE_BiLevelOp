from unittest import result
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.termination import get_termination

from GE import Grammar, mapping_breadth_first, mapping_pigrammatical, mapping_depth_first
import multiprocessing as mp
from os.path import join
from copy import deepcopy
import numpy as np
from SR import SR
# from MathFunctions import pdiv, plog, psqrt

from PermOp import *
import matplotlib.pyplot as plt


def evaluate_sr(grammar, sr, x, order=None):
    expr = grammar.mapping(x, order)
    return sr.evaluate(expr)


class SRProblem(Problem):
    def __init__(self, n_var=-1, xl=None, xu=None, grammar=None, instance=None, **kwargs):
        super().__init__(n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, type_var=np.int_, elementwise_evaluation=False,**kwargs)
        self.grammar = deepcopy(grammar)
        self.sr_problem = deepcopy(instance)
        self.values = np.random.permutation(n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.asarray([evaluate_sr(self.grammar, self.sr_problem, self.values, item) for item in x])


if __name__ == "__main__":
    instance_path = "instances"
    instances = ["Keijzer6"]  # "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"

    mappings = [mapping_breadth_first, mapping_depth_first]

    val = []
    for mapping in mappings:
        for instance in instances[:2]:
            sr_train = SR(file=join(instance_path, instance, "train.csv"))
            sr_test = SR(file=join(instance_path, instance, "test.csv"))
            
            gramatica = Grammar(mapping=mapping, file=join(instance_path, instance, "grammar.bnf"))
            problema = SRProblem(200, xl=0.0, xu=15.0, grammar=gramatica,
                                 instance=sr_train)

            algoritmo = GA(
                pop_size=500,
                eliminate_duplicates=True,
                sampling=PermutationRandomSampling(),
                crossover=PMX(),
                mutation=OpMutation(),
            )
            # ("op", OpMutation, dict(size=3))
            termination = get_termination("n_gen", 10) # ("n_eval", 25000) # get_termination("n_gen", 1000)

            res = minimize(problema,
                        algoritmo,
                        termination,
                        save_history=True,
                        verbose=False)

            print("-"*40)
            print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
            results = gramatica.mapping(problema.values, res.X)
            print(mapping.__name__, instance,results)
            print(res.F, sr_test.evaluate(results))
            
            val = [e.opt.get("F")[0][0] for e in res.history]
            plt.plot(np.arange(len(val)), val, label=f'{mapping.__name__}, {instance}')
    plt.legend()
    plt.show()
