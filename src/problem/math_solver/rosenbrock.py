"""
Parametric Mixed Integer Constrained Rosenbrock Problem
"""

import numpy as np
from pyomo import environ as pe

from src.problem.math_solver import abcParamSolver

class rosenbrock(abcParamSolver):
    def __init__(self, steepness, num_blocks, timelimit=None):
        super().__init__(timelimit=timelimit)
        # create model
        m = pe.ConcreteModel()
        # parameters
        m.p = pe.Param(default=1, mutable=True)
        m.a = pe.Param(pe.RangeSet(0, num_blocks-1), default=1, mutable=True)
        # variables
        m.x = pe.Var(pe.RangeSet(0, num_blocks*2-1), domain=pe.Reals)
        for i in range(num_blocks):
            # integer variables
            m.x[2*i+1].domain = pe.Integers
        # objective
        obj = sum((m.a[i] - m.x[2*i]) ** 2 + \
                   steepness * (m.x[2*i+1] - m.x[2*i] ** 2) ** 2 for i in range(num_blocks))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # constraints
        m.cons = pe.ConstraintList()
        m.cons.add(sum(m.x[2*i+1] for i in range(num_blocks)) >= num_blocks * m.p / 2)
        m.cons.add(sum(m.x[2*i] ** 2 for i in range(num_blocks)) <= num_blocks * m.p)
        rng = np.random.RandomState(17)
        b = rng.normal(scale=1, size=(num_blocks))
        q = rng.normal(scale=1, size=(num_blocks))
        m.cons.add(sum(b[i] * m.x[2*i] for i in range(num_blocks)) <= 0)
        m.cons.add(sum(q[i] * m.x[2*i+1] for i in range(num_blocks)) <= 0)
        # attribute
        self.model = m
        self.params ={"p":m.p, "a":m.a}
        self.vars = {"x":m.x}
        self.cons = m.cons

if __name__ == "__main__":

    from src.utils import ms_test_solve

    steepness = 50    # steepness factor
    num_blocks = 10   # number of expression blocks
    timelimit = 1000    # time limit

    # params
    p, a = 3.2, (2.4, 1.8)
    params = {"p":p, "a":a}
    # init model
    model = rosenbrock(steepness=steepness, num_blocks=num_blocks, timelimit=timelimit)

    # solve the MIQP
    print("======================================================")
    print("Solve MINLP problem:")
    model.set_param_val(params)
    solvals, _ = ms_test_solve(model, tee=True)

    # warm starting
    print()
    print("======================================================")
    print("Warm start:")
    model.set_param_val(params)
    model.set_warm_start(solvals)
    ms_test_solve(model, tee=True)

    # solve the penalty
    print()
    print("======================================================")
    print("Solve penalty problem:")
    model_pen = model.penalty(100)
    model_pen.set_param_val(params)
    # scip
    ms_test_solve(model_pen)

    # solve the relaxation
    print()
    print("======================================================")
    print("Solve relaxed problem:")
    model_rel = model.relax()
    model_rel.set_param_val(params)
    # scip
    ms_test_solve(model_rel)
