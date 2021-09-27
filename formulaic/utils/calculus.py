from typing import Iterable, Set

from formulaic.parser.types import Factor, Term


def differentiate_term(
    term: Term, vars: Iterable[str], use_sympy: bool = False
) -> Term:
    """
    Symbolically differentiate a `Term` instance with respect to one or more `vars`.

    Args:
        term: The `Term` instance to differentiate.
        vars: The variables by which to differentiate.
        use_sympy: Whether to interpret factor token strings using sympy. If
            `True`, symbolic factors like `log(x)` can be differentiated with
            respect to `x`. If `False`, factor token strings must match the
            variable exactly in order to be detected.

    Returns:
        A new `Term` instance representing the differentiated term.

    Notes:
        - This method takes into account the chain rule/etc.
        - Care must be taken to make sure that the symbolic representation of
            the factors can be properly interpreted by `sympy`. For example, `I(x)`
            would not be understood.
    """
    factors = term.factors

    for var in vars:
        affected_factors = set(
            factor
            for factor in factors
            if var in _factor_symbols(factor, use_sympy=use_sympy)
        )
        if not affected_factors:
            return Term({Factor("0", eval_method="literal")})
        factors = factors.difference(affected_factors).union(
            _differentiate_factors(affected_factors, var, use_sympy=use_sympy)
        )

    return Term(factors or {Factor("1", eval_method="literal")})


def _factor_symbols(factor: Factor, use_sympy: bool = False) -> Set[str]:
    """
    Extract the symbols represented in a factor.

    Args:
        factor: The `Factor` instance from which symbols should be extracted.
        use_sympy: Whether to interpret the string representation of the
            factor using `sympy`.

    Returns:
        The set of string symbols represented by the factor.
    """
    if use_sympy:
        try:
            import sympy

            return {str(s) for s in sympy.S(factor.expr).free_symbols}
        except ImportError:  # pragma: no cover
            raise ImportError(
                "`sympy` is not available. Install it using `pip install formulaic[calculus]` or `pip install sympy`."
            )
    return {factor.expr}


def _differentiate_factors(
    factors: Set[Factor], var: str, use_sympy: bool = False
) -> Set[Factor]:
    """
    Differentiate the nominated `factors` by `var`.

    Args:
        factors: The set of factors which should be differentiated (taking for
            granted that they are multiplied together).
        var: The variable by which to differentiate.
        use_sympy: Whether to perform the differentiation using sympy, allowing
            for symbolic differentiations like `log(x)` -> `1/x`.

    Returns:
        A set containing the new factors to replace the incoming factors in a
        term.
    """
    if use_sympy:
        try:
            import sympy

            expr = sympy.S(
                "(" + ") * (".join(factor.expr for factor in factors) + ")"
            ).diff(var)
            eval_method = "python"
        except ImportError:  # pragma: no cover
            raise ImportError(
                "`sympy` is not available. Install it using `pip install formulaic[calculus]` or `pip install sympy`."
            )
    else:
        assert len(factors) == 1
        expr = 1
        eval_method = next(iter(factors)).eval_method

    if expr == 1:
        return set()
    return {Factor(f"({str(expr)})", eval_method=eval_method)}
