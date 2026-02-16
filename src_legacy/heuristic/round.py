import math

def naive_round(xval_rel, model):
    """
    A method to just round
    """
    # get solution value
    for k, vals in xval_rel.items():
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

def floor_round(xval_rel, model):
    """
    A method to just round
    """
    # get solution value
    for k, vals in xval_rel.items():
        # assign initial solution
        for i in vals:
            # round integer variables
            if model.vars[k][i].is_integer():
                model.vars[k][i].value = math.floor(vals[i])
            # assign continuous variables
            else:
                model.vars[k][i].value = vals[i]
    xval, objval = model.get_val()
    return xval, objval
