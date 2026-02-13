"""
Test functions to solve problem
"""

import time
import torch

def ms_test_solve(model, tee=False):
    """
    Test function for mathematic solver
    """
    tick = time.time()
    solvals, objval = model.solve(tee=tee)
    tock = time.time()
    print("Binary Variables:", model.bin_ind)
    print("Integer Variables:", model.int_ind)
    for k, v in solvals.items():
        for i, val in v.items():
            print("{}[{}]: {:.2f}".format(k, i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.cal_violation())))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()
    return solvals, objval


def nm_test_solve(var_key, components, datapoint, model):
    """
    Test function for neuroMANCER
    """
    # inference
    components.eval()
    tick = time.time()
    with torch.no_grad():
        for comp in components:
            datapoint.update(comp(datapoint))
    tock = time.time()
    # get values
    for i in model.vars["x"]:
        model.vars["x"][i].value = datapoint[var_key][0, i].item()
    solvals, objval = model.get_val()
    # results
    print("Binary Variables:", model.bin_ind)
    print("Integer Variables:", model.int_ind)
    for i, val in solvals["x"].items():
        print("{}[{}]: {:.2f}".format(var_key, i, val), end=" ")
    print("\nObjective Value: {:.2f}".format(objval))
    print("Constraint Violations: {:.4f}".format(sum(model.cal_violation())))
    print("Elapsed Time: {:.4f} sec".format(tock - tick))
    print()
    return solvals, objval
