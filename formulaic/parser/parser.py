import itertools
import functools

from .algos.infix_to_ast import infix_to_ast
from .algos.tokenize import tokenize
from .types import Token, Operator


class FormulaParser:

    @property
    def operators(self):
        return [
            Operator("~", arity=2, precedence=-100, associativity=None, to_terms=lambda lhs, rhs: (lhs.to_terms(), rhs.to_terms())),
            Operator("+", arity=2, precedence=100, associativity='left', to_terms=lambda *args: set(itertools.chain(*[arg.to_terms() for arg in args]))),
            Operator("-", arity=2, precedence=100, associativity='left', to_terms=lambda left, right: set(set(left.to_terms()).difference(right.to_terms()))),
            Operator("*", arity=2, precedence=200, associativity='left', to_terms=lambda *args: (
                {
                    functools.reduce(lambda x, y: x * y, term)
                    for term in itertools.product(*[arg.to_terms() for arg in args])
                }
                .union(itertools.chain(*[arg.to_terms() for arg in args]))
            )),
            Operator("/", arity=2, precedence=200, associativity='left'),
            Operator(":", arity=2, precedence=300, associativity='left', to_terms=lambda *args: {
                functools.reduce(lambda x, y: x * y, term)
                for term in itertools.product(*[arg.to_terms() for arg in args])
            }),
            Operator("**", arity=2, precedence=500, associativity='right'),

            Operator("+", arity=1, precedence=1000, associativity='right', fixity='prefix'),
            Operator("-", arity=1, precedence=1000, associativity='right', fixity='prefix'),
        ]

    def __call__(self, formula):
        return self.get_ast(formula).to_terms()

    def get_tokens(self, formula):
        tokens = list(tokenize(formula))

        # Insert "1" or "1 + " to beginning of RHS formula
        one = Token(token='1', kind='value')
        plus = Token(token='+', kind='operator')
        minus = Token(token='-', kind='operator')

        if len(tokens) == 0:
            tokens = [one]
        else:
            try:
                tilde_index = tokens.index('~')
                if tilde_index == len(tokens) - 1:
                    tokens.append(one)
                else:
                    tokens.insert(tilde_index+1, one)
                    tokens.insert(tilde_index+2, plus)
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

    def get_ast(self, formula):
        return infix_to_ast(self.get_tokens(formula), operators=self.operators)
