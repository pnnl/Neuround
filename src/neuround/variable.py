"""
Variable type management and unified variable creation for NeuroPMINLP.
"""

from enum import Enum

import neuromancer as nm


class VarType(Enum):
    """
    Variable type enumeration.
    """
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"

    def __repr__(self):
        return f"VarType.{self.name}"


def _build_var_types(num_vars, integer_indices=None, binary_indices=None):
    """Build a VarType list from num_vars and index lists."""
    # Check for overlap between integer and binary indices
    overlap = set(integer_indices or []) & set(binary_indices or [])
    if overlap:
        raise ValueError(
            f"Indices {overlap} appear in both integer_indices and "
            f"binary_indices. Each index must have exactly one type."
        )

    # Initialize all as continuous as default
    types = [VarType.CONTINUOUS] * num_vars

    # Set integer types based on indices
    for i in (integer_indices or []):
        if not 0 <= i < num_vars:
            raise ValueError(
                f"Integer index {i} out of range [0, {num_vars})"
            )
        types[i] = VarType.INTEGER

    # Set binary types based on indices
    for i in (binary_indices or []):
        if not 0 <= i < num_vars:
            raise ValueError(
                f"Binary index {i} out of range [0, {num_vars})"
            )
        types[i] = VarType.BINARY

    return types


def _resolve_var_types(num_vars, integer_indices, binary_indices, var_types):
    """Dispatch to explicit list or index-based construction."""
    # Resolve types
    if var_types is not None:
        if (num_vars is not None or
                integer_indices is not None or
                binary_indices is not None):
            raise ValueError(
                "Cannot specify both var_types and indices-based parameters. "
                "Choose one approach: either pass var_types list OR use indices."
            )
        return list(var_types)

    # If no explicit var_types, build from indices
    if num_vars is None:
        raise ValueError(
            "num_vars is required when using integer_indices or binary_indices."
        )
    return _build_var_types(num_vars, integer_indices, binary_indices)


def _attach_metadata(var, types):
    """Attach type metadata (var_types, indices) to var."""
    # Attach variable types
    var.var_types = types
    # Attach number of variables
    var.num_vars = len(types)
    # Attach indices for each type
    var.integer_indices = [i for i, vt in enumerate(types)
                           if vt == VarType.INTEGER]
    var.binary_indices = [i for i, vt in enumerate(types)
                          if vt == VarType.BINARY]
    var.continuous_indices = [i for i, vt in enumerate(types)
                              if vt == VarType.CONTINUOUS]


def _attach_relaxed(var):
    """Attach relaxed variable (key + "_rel") if discrete vars exist."""
    # If there are discrete variables
    has_discrete = len(var.integer_indices) > 0 or len(var.binary_indices) > 0
    # Create new relaxed variable with "_rel" suffix
    if has_discrete:
        var.relaxed = nm.variable(var.key + "_rel")
    # Relaxed version is itself
    else:
        var.relaxed = var


def variable(key, num_vars=None,
             integer_indices=None,
             binary_indices=None,
             var_types=None):
    """
    Create a neuromancer variable, optionally with type metadata.

    Without type params: equivalent to ``nm.variable(key)``.
    With type params: attaches ``var_types``, ``num_vars``, indices,
    and ``relaxed`` (continuous relaxation, key = key + "_rel").

    Args:
        key: Variable name.
        num_vars: Total number of variables.
        integer_indices: Indices of integer variables.
        binary_indices: Indices of binary variables.
        var_types: Explicit VarType list (mutually exclusive with indices).

    Returns:
        Neuromancer variable, with type attrs attached if type params given.
    """
    # Create base variable
    var = nm.variable(key)

    # No type information -> plain neuromancer variable
    if num_vars is None and var_types is None:
        if integer_indices is not None or binary_indices is not None:
            raise ValueError(
                "num_vars is required when using integer_indices or binary_indices."
            )
        return var

    # Attach type metadata
    types = _resolve_var_types(num_vars, 
                               integer_indices, 
                               binary_indices,
                               var_types)
    _attach_metadata(var, types)

    # Attach relaxed variable if discrete variables exist
    _attach_relaxed(var)

    return var
