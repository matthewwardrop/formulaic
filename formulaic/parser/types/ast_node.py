class ASTNode:

    def __init__(self, operator, args):
        self.operator = operator
        self.args = args

    def to_terms(self):
        return self.operator.to_terms(*self.args)

    def __repr__(self):
        return f"<ASTNode {self.operator}: {self.args}>"
