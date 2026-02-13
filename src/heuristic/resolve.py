import math

from pyomo import environ as pe

def rens(xval_rel, model, tolerance=1e-5):
    # clone model
    model_rens = model.clone()
    # get solution value
    for k, vals in xval_rel.items():
        # go through vars
        for i in vals:
            # skip continuous
            if model_rens.vars[k][i].domain == pe.Reals:
                continue
            # fix integer variable in relaxation
            if abs(xval_rel[k][i] - round(xval_rel[k][i])) <= tolerance:
                model_rens.vars[k][i].fix(round(xval_rel[k][i]))
            # change to binary bounds
            else:
                model_rens.vars[k][i].setlb(math.floor(xval_rel[k][i]))
                model_rens.vars[k][i].setub(math.ceil(xval_rel[k][i]))
    # solve
    xval, objval = model_rens.solve()
    # assign solution value
    for k, vals in xval.items():
        # assign initial solution
        for i in vals:
            # round integer variables
            if model.vars[k][i].is_integer():
                model.vars[k][i].value = round(vals[i])
            # assign continuous variables
            else:
                model.vars[k][i].value = vals[i]
    xval, objval = model.get_val()
    return xval, objval

if __name__ == "__main__":

    from src.problem import msRosenbrock

    steepness = 50    # steepness factor
    num_blocks = 2    # number of expression blocks
    timelimit = 60    # time limit

    # init model
    model = msRosenbrock(steepness=steepness, num_blocks=num_blocks, timelimit=timelimit)

    # params
    p, a = 3.2, (2.4, 1.8)
    params = {"p":p, "a":a}
    # set params
    model.set_param_val({"p":p, "a":a})
    # relax model
    model_rel = model.relax()
    # solve relaxation
    xval_rel, _ = model_rel.solve()
    print(xval_rel)
    # rens
    xval, objval = rens(xval_rel, model)
