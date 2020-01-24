import itertools
import functools

from formulaic.errors import FormulaParsingError

from .algos.infix_to_ast import infix_to_ast
from .algos.tokenize import tokenize
from .types import Factor, Term, Token, Operator


class FormulaParser:

    @property
    def operators(self):

        def formula_separator_expansion(lhs, rhs):
            terms = (lhs.to_terms(), rhs.to_terms())

            out = []
            for termset in terms:
                if isinstance(termset, tuple):
                    out.extend(termset)
                else:
                    out.append(termset)
            return tuple(out)

        def nested_product_expansion(parents, nested):
            terms = parents.to_terms()
            common = functools.reduce(lambda x, y: x * y, terms)
            return terms.union({
                common * term
                for term in nested.to_terms()
            })

        def unary_negation(arg):
            terms = arg.to_terms()
            if len(terms) > 1 or list(terms)[0] != '0':
                raise FormulaParsingError("Unary negation is only implemented for '0', where it is substituted for '1'.")
            return {Term(factors=[Factor('1', eval_method='literal')])}

        return [
            Operator("~", arity=2, precedence=-100, associativity=None, to_terms=formula_separator_expansion),
            Operator("~", arity=1, precedence=-100, associativity=None, fixity='prefix', to_terms=lambda expr: (expr.to_terms(), )),
            Operator("+", arity=2, precedence=100, associativity='left', to_terms=lambda *args: set(itertools.chain(*[arg.to_terms() for arg in args]))),
            Operator("-", arity=2, precedence=100, associativity='left', to_terms=lambda left, right: set(set(left.to_terms()).difference(right.to_terms()))),
            Operator("+", arity=1, precedence=100, associativity='right', fixity='prefix', to_terms=lambda arg: arg.to_terms()),
            Operator("-", arity=1, precedence=100, associativity='right', fixity='prefix', to_terms=unary_negation),
            Operator("*", arity=2, precedence=200, associativity='left', to_terms=lambda *args: (
                {
                    functools.reduce(lambda x, y: x * y, term)
                    for term in itertools.product(*[arg.to_terms() for arg in args])
                }
                .union(itertools.chain(*[arg.to_terms() for arg in args]))
            )),
            Operator("/", arity=2, precedence=200, associativity='left', to_terms=nested_product_expansion),
            Operator(":", arity=2, precedence=300, associativity='left', to_terms=lambda *args: {
                functools.reduce(lambda x, y: x * y, term)
                for term in itertools.product(*[arg.to_terms() for arg in args])
            }),
            Operator("**", arity=2, precedence=500, associativity='right', to_terms=lambda arg, power: {
                functools.reduce(lambda x, y: x * y, term)
                for term in itertools.product(*[arg.to_terms()] * int(power.token))
            }),
        ]

    def get_tokens(self, formula, *, include_intercept=True):
        tokens = list(tokenize(formula))

        # Insert "1" or "1 + " to beginning of RHS formula
        one = Token(token='1', kind='value')
        plus = Token(token='+', kind='operator')
        minus = Token(token='-', kind='operator')

        if include_intercept:
            if len(tokens) == 0:
                tokens = [one]
            else:
                try:
                    tilde_index = len(tokens) - 1 - tokens[::-1].index('~')
                    if tilde_index == len(tokens) - 1:
                        tokens.append(one)
                    else:
                        tokens.insert(tilde_index + 1, one)
                        tokens.insert(tilde_index + 2, plus)
                except ValueError:
                    tokens.insert(0, one)
                    tokens.insert(1, plus)

        # Replace "0" with "-1"
        try:
            zero_index = tokens.index('0')
            if tokens[zero_index - 1] == '+':
                tokens[zero_index - 1] = minus
                tokens[zero_index] = one
        except ValueError:
            pass

        return tokens

    def get_ast(self, formula, *, include_intercept=True):
        return infix_to_ast(self.get_tokens(formula, include_intercept=include_intercept), operators=self.operators)

    def get_terms(self, formula, *, sort=True, include_intercept=True):
        terms = self.get_ast(formula, include_intercept=include_intercept).to_terms()

        if sort:
            if isinstance(terms, tuple):
                terms = tuple(sorted(ts) for ts in terms)
            else:
                terms = sorted(terms)

        return terms
