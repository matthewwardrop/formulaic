class ASTNode:

    def __init__(self, operator, args):
        self.operator = operator
        self.args = args

    def to_terms(self):
        return self.operator.to_terms(*self.args)

    def __repr__(self):
        return f"<ASTNode {self.operator}: {self.args}>"

    def flatten(self, str_args=False):
        return [
            str(self.operator) if str_args else self.operator,
            *[
                arg.flatten(str_args=str_args) if isinstance(arg, ASTNode) else (str(arg) if str_args else arg)
                for arg in self.args
            ]
        ]
