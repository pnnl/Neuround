"""
Parametric Mixed Integer Quadratic Programming

https://arxiv.org/abs/2104.12225
"""

import numpy as np
from pyomo import environ as pe

from experiments.math_solver.abc_solver import abcParamSolver

class quadratic(abcParamSolver):
    def __init__(self, num_var, num_ineq, timelimit=None):
        super().__init__(timelimit=timelimit, solver="gurobi")
        # Fixed parameters
        rng = np.random.RandomState(17)
        Q = 0.01 * np.diag(rng.random(size=num_var))
        p = 0.1 * rng.random(num_var)
        A = rng.normal(scale=0.1, size=(num_ineq, num_var))
        # Create model
        m = pe.ConcreteModel()
        # Mutable parameters (parametric part of the problem)
        m.b = pe.Param(pe.RangeSet(0, num_ineq-1), default=0, mutable=True)
        # Decision variables
        m.x = pe.Var(range(num_var), domain=pe.Integers)
        # Objective function: 1/2 x^T Q x + p^T x
        obj = sum(m.x[j] * Q[j,j] * m.x[j] / 2 + p[j] * m.x[j] for j in range(num_var))
        m.obj = pe.Objective(sense=pe.minimize, expr=obj)
        # Constraints: A x <= b
        m.cons = pe.ConstraintList()
        for i in range(num_ineq):
            m.cons.add(sum(A[i,j] * m.x[j] for j in range(num_var)) <= m.b[i])
        # Set attributes
        self.model = m
        self.params = {"b": m.b}
        self.vars = {"x": m.x}
        self.cons = m.cons
