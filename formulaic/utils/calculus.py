from formulaic.parser.types import Factor, Term


def differentiate_term(term, vars, use_sympy=False):
    factors = term.factors

    for var in vars:
        affected_factors = set(
            factor
            for factor in factors
            if var in _factor_symbols(factor, use_sympy=use_sympy)
        )
        if not affected_factors:
            return Term({Factor('0', eval_method='literal')})
        factors = factors.difference(affected_factors).union(_differentiate_factors(affected_factors, var, use_sympy=use_sympy))

    return Term(factors or {Factor('1', eval_method='literal')})


def _factor_symbols(factor, use_sympy=False):
    if use_sympy:
        try:
            import sympy
            return {str(s) for s in sympy.S(factor.expr).free_symbols}
        except ImportError:  # pragma: no cover
            raise ImportError("`sympy` is not available. Install it using `pip install formulaic[calculus]` or `pip install sympy`.")
    return {factor.expr}


def _differentiate_factors(factors, var, use_sympy=False):
    if use_sympy:
        try:
            import sympy
            expr = sympy.S('(' + ') * ('.join(factor.expr for factor in factors) + ')').diff(var)
            eval_method = 'python'
        except ImportError:  # pragma: no cover
            raise ImportError("`sympy` is not available. Install it using `pip install formulaic[calculus]` or `pip install sympy`.")
    else:
        assert len(factors) == 1
        expr = 1
        eval_method = next(iter(factors)).eval_method

    if expr == 1:
        return set()
    return {Factor(f'({str(expr)})', eval_method=eval_method)}
