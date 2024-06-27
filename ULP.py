from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling, PermutationRandomSampling
from pymoo.termination import get_termination
from GE import Grammar, mapping_depth_first, mapping_breadth_first, mapping_pigrammatical
from os.path import join
from copy import deepcopy
import numpy as np
from SR import SR


def evaluate_sr(grammar, sr, x, order=None):
    expr = grammar.mapping(x, order)
    return sr.evaluate(expr)


class UpperLevelProblem(Problem):
    def __init__(self, llp, n_var=-1, xl=None, xu=None, grammar=None, instance=None, **kwargs):
        super().__init__(n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, type_var=np.int_, elementwise_evaluation=False, **kwargs)
        self.grammar = deepcopy(grammar)
        self.sr_problem = deepcopy(instance)
        self.llp = llp

    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate the lower level problem
        out["F"] = np.asarray([evaluate_sr(self.grammar, self.sr_problem, item) for item in x])


class LowerLevelProblem(Problem):
    def __init__(self, n_var=-1, xl=None, xu=None, grammar=None, instance=None, **kwargs):
        super().__init__(n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, type_var=np.int_, elementwise_evaluation=False,
                         **kwargs)
        self.grammar = deepcopy(grammar)
        self.sr_problem = deepcopy(instance)
        self.values = np.random.permutation(n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.asarray([evaluate_sr(self.grammar, self.sr_problem, self.values, item) for item in x])


if __name__ == "__main__":
    instance_path = "instances"
    instances = ["F1"]  # "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "Keijzer6"

    mappings = [mapping_depth_first]  # mapping_breadth_first, mapping_depth_first

    val = []
    for mapping in mappings:
        for instance in instances[:2]:
            sr_train = SR(file=join(instance_path, instance, "train.csv"))
            sr_test = SR(file=join(instance_path, instance, "test.csv"))

            grammarP = Grammar(mapping=mapping, file=join(instance_path, instance, "grammar.bnf"))

            llp = LowerLevelProblem(n_var=100, xl=0.0, xu=15.0, grammar=grammarP, instance=sr_train)

            ulp = UpperLevelProblem(llp, n_var=100, xl=0.0, xu=255.0, grammar=grammarP,
                                 instance=sr_train)

            algorithm = GA(
                pop_size=300,
                eliminate_duplicates=True,
                sampling=IntegerRandomSampling(),
                crossover=TwoPointCrossover(),
                mutation=BitflipMutation(),
            )

            termination = get_termination("n_eval", 25000)

            res = minimize(ulp,
                           algorithm,
                           termination,
                           save_history=True,
                           verbose=False)

            print("-" * 40)
            print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
            results = grammarP.mapping(res.X)
            print(mapping.__name__, instance, results)
            print(res.F, sr_test.evaluate(results))
