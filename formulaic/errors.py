# Top-level error and warning classes


class FormulaicError(Exception):
    pass


class FormulaicWarning(Warning):
    pass


# Formula parsing errors


class FormulaInvalidError(FormulaicError):
    """
    Provided formula specification is not a valid format.
    """


class FormulaParsingError(FormulaicError):
    """
    An error occured during the parsing of a formula specification.
    """


class FormulaSyntaxError(FormulaParsingError):
    """
    Could not tokenize the nominated formula specification.
    """


# Formula materializer meta-errors


class FormulaMaterializerInvalidError(FormulaicError):
    pass


class FormulaMaterializerNotFoundError(FormulaicError):
    pass


# Data materialization errors and warnings


class FormulaMaterializationError(FormulaicError):
    pass


class FactorEncodingError(FormulaMaterializationError):
    pass


class FactorEvaluationError(FormulaMaterializationError):
    pass


class DataMismatchWarning(FormulaicWarning):
    pass
