# Top-level error and warning classes


class FormulaicError(Exception):
    pass


class FormulaicWarning(Warning):
    pass


# Formula parsing errors


class FormulaInvalidError(FormulaicError):
    pass


class FormulaParsingError(FormulaicError):
    pass


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
