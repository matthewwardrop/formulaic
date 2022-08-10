from __future__ import annotations
from abc import abstractmethod

import inspect
import warnings
from numbers import Number
from typing import Any, Union, Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy
import pandas
import scipy.sparse as spsparse
import scipy.sparse.linalg
from interface_meta import InterfaceMeta

from formulaic.errors import DataMismatchWarning
from formulaic.materializers.types import FactorValues
from formulaic.utils.sparse import categorical_encode_series_to_sparse_csc_matrix
from formulaic.utils.stateful_transforms import stateful_transform

from .poly import poly

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec  # pragma: no cover


def C(
    data: Any,
    contrasts: Optional[
        Union[Contrasts, Dict[str, Iterable[Number]], numpy.ndarray]
    ] = None,
    *,
    levels: Optional[Iterable[str]] = None,
):
    """
    Mark data as being categorical, and optionally specify the contrasts to be
    used during encoding.

    Args:
        data: The data to be marked as categorical.
        contrasts:  The specification of the contrasts that are to be computed.
            Should be a `Constrasts` instance, a dictionary mapping a key for
            the contrast with a vector of weights for the categories, or a
            numpy array with columns representing the constrasts, and rows
            representing the weights over the categories in the data. If not
            specified, a `Treatment` encoding is assumed.
        levels: The categorical levels associated with `data`. If not present,
            levels are inferred from `data`. Note that extra levels in `data`
            will be treated as null data.
    """

    def encoder(
        values: Any,
        reduced_rank: bool,
        drop_rows: List[int],
        encoder_state: Dict[str, Any],
        model_spec: ModelSpec,
    ):
        values = pandas.Series(values)
        values = values.drop(index=values.index[drop_rows])
        return encode_contrasts(
            values,
            contrasts=contrasts,
            levels=levels,
            reduced_rank=reduced_rank,
            _state=encoder_state,
            _spec=model_spec,
        )

    return FactorValues(
        data,
        kind="categorical",
        spans_intercept=True,
        encoder=encoder,
    )


@stateful_transform
def encode_contrasts(
    data,
    contrasts: Union[
        Contrasts, Dict[str, Iterable[Number]], numpy.ndarray, None
    ] = None,
    *,
    levels: Optional[Iterable[str]] = None,
    reduced_rank: bool = False,
    output: Optional[str] = None,
    _state=None,
    _spec=None,
) -> FactorValues[Union[pandas.DataFrame, spsparse.spmatrix]]:
    """
    Encode a categorical dataset into one or more "constrasts".

    Args:
        data: The categorical data array/series to be encoded.
        contrasts: The specification of the contrasts that are to be computed.
            Should be a `Constrasts` instance, a dictionary mapping a key for
            the contrast with a vector of weights for the categories, or a
            numpy array with columns representing the constrasts, and rows
            representing the weights over the categories in the data. If not
            specified, a `Treatment` encoding is assumed.
        levels: The complete set of levels (categories) posited to be present in
            the data. This can also be used to reorder the levels as needed.
        reduced_rank: Whether to reduce the rank of output encoded columns in
            order to avoid spanning the intercept.
        output: The type of data to output. Must be one of "pandas", "numpy", or
            "sparse".
    """
    # Prepare arguments
    output = output or _spec.output or "pandas"
    levels = levels or _state.get(
        "categories"
    )  # TODO: Is this too early to provide useful feedback to users?

    if contrasts is None:
        contrasts = TreatmentContrasts()
    elif inspect.isclass(contrasts) and issubclass(contrasts, Contrasts):
        contrasts = contrasts()
    if not isinstance(contrasts, Contrasts):
        contrasts = CustomContrasts(contrasts)

    if levels is not None:
        extra_categories = set(pandas.unique(data)).difference(levels)
        if extra_categories:
            warnings.warn(
                "Data has categories outside of the nominated levels (or that were "
                f"not seen in original dataset): {extra_categories}. They are being "
                " cast to nan, which will likely skew the results of your analyses.",
                DataMismatchWarning,
            )
        data = pandas.Series(pandas.Categorical(data, categories=levels))
    else:
        data = pandas.Series(data).astype("category")

    # Perform dummy encoding
    if output in ("pandas", "numpy"):
        categories = list(data.cat.categories)
        encoded = pandas.get_dummies(data)
    elif output == "sparse":
        categories, encoded = categorical_encode_series_to_sparse_csc_matrix(
            data,
        )
    else:
        raise ValueError(f"Unknown output type `{repr(output)}`.")

    # Update state
    _state["categories"] = categories

    # Apply and return contrasts
    return contrasts.apply(
        encoded, levels=categories, reduced_rank=reduced_rank, output=output
    )


class Contrasts(metaclass=InterfaceMeta):
    """
    The base class for all contrast implementations.
    """

    INTERFACE_RAISE_ON_VIOLATION = True

    FACTOR_FORMAT = "{name}[{field}]"

    def apply(
        self,
        dummies,
        levels,
        reduced_rank=True,
        output: Optional[str] = None,
    ):
        """
        Apply the contrasts defined by this `Constrasts` instance to `dummies`
        (the dummy encoding of the values of interest).

        Args:
            dummies: Dummy encoded representation of the values.
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether to output a reduced rank matrix. When this is
                `False`, the dummy encoding is usually passed through
                unmodified.
            output: The type of datastructure to output. Should be one of:
                "pandas", "numpy", "sparse", or `None`. If `None` is provided,
                the output type will be inferred from the input data type.
        """

        if output is None:
            if isinstance(dummies, pandas.DataFrame):
                output = "pandas"
            elif isinstance(dummies, numpy.ndarray):
                output = "numpy"
            elif isinstance(dummies, spsparse.spmatrix):
                output = "sparse"
            else:  # pragma: no cover
                raise ValueError(
                    f"Cannot impute output type for dummies of type `{type(dummies)}`."
                )
        elif output not in ("pandas", "numpy", "sparse"):  # pragma: no cover
            raise ValueError(
                "Output type for contrasts must be one of: 'pandas', 'numpy' or 'sparse'."
            )

        sparse = output == "sparse"
        encoded = self._apply(
            dummies, levels=levels, reduced_rank=reduced_rank, sparse=sparse
        )
        coding_column_names = self.get_coding_column_names(
            levels, reduced_rank=reduced_rank
        )

        if output == "pandas":
            encoded = pandas.DataFrame(
                encoded,
                columns=coding_column_names,
            )
        elif output == "numpy":
            encoded = numpy.array(encoded)

        return FactorValues(
            encoded,
            kind="categorical",
            column_names=coding_column_names,
            spans_intercept=self.get_spans_intercept(levels, reduced_rank=reduced_rank),
            drop_field=self.get_drop_field(levels, reduced_rank=reduced_rank),
            format=self.get_factor_format(levels, reduced_rank=reduced_rank),
            encoded=True,
        )

    def _apply(self, dummies, levels, reduced_rank=True, sparse=False):
        coding_matrix = self.get_coding_matrix(levels, reduced_rank, sparse=sparse)
        return (dummies if sparse else dummies.values) @ coding_matrix

    # Coding matrix methods

    def get_coding_matrix(self, levels, reduced_rank=True, sparse=False):
        """
        Generate the coding matrix; i.e. the matrix with column vectors
        representing the encoding to use for the corresponding level.

        Args:
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether to output a reduced rank matrix. When this is
                `False`, the dummy encoding is usually passed through
                unmodified.
            sparse: Whether to output sparse results.
        """
        coding_matrix = self._get_coding_matrix(
            levels, reduced_rank=reduced_rank, sparse=sparse
        )

        if sparse:
            return coding_matrix

        return pandas.DataFrame(
            coding_matrix,
            columns=self.get_coding_column_names(levels, reduced_rank=reduced_rank),
            index=levels,
        )

    @abstractmethod
    def _get_coding_matrix(self, levels, reduced_rank=True, sparse=False):
        """
        Subclasses must override this method to implement the generation of the
        coding matrix.

        Args:
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether to output the reduced rank coding matrix.
            sparse: Whether to output sparse results.
        """

    @abstractmethod
    def get_coding_column_names(self, levels, reduced_rank=True):
        """
        Generate the names for the columns of the coding matrix (the encoded
        features to be added to the model matrix).

        Args:
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether to output the coefficients for reduced rank
                encodings.
        """

    # Coefficient matrix methods

    def get_coefficient_matrix(self, levels, reduced_rank=True, sparse=False):
        """
        Generate the coefficient matrix; i.e. the matrix with rows representing
        the contrasts effectively computed during a regression, with columns
        indicating the weights given to the origin categories. This is primarily
        used for debugging/introspection.

        Args:
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether to output the coefficients for reduced rank
                encodings.
            sparse: Whether to output sparse results.
        """
        coefficient_matrix = self._get_coefficient_matrix(
            levels, reduced_rank=reduced_rank, sparse=sparse
        )
        if sparse:
            return coefficient_matrix
        return pandas.DataFrame(
            coefficient_matrix,
            columns=levels,
            index=self.get_coefficient_row_names(levels, reduced_rank=reduced_rank),
        )

    def _get_coefficient_matrix(self, levels, reduced_rank=True, sparse=False):
        coding_matrix = self.get_coding_matrix(
            levels, reduced_rank=reduced_rank, sparse=sparse
        )

        if reduced_rank:
            coding_matrix = (spsparse if sparse else numpy).hstack(
                [
                    numpy.ones((len(levels), 1)),
                    coding_matrix,
                ]
            )

        if sparse:
            return scipy.sparse.linalg.inv(coding_matrix.tocsc())
        return numpy.linalg.inv(coding_matrix)

    @abstractmethod
    def get_coefficient_row_names(self, levels, reduced_rank=True):
        """
        Generate the names for the rows of the coefficient matrix (the
        interpretation of the contrasts generated by the coding matrix).

        Args:
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether to output the coefficients for reduced rank
                encodings.
        """

    # Additional metadata

    def get_spans_intercept(self, levels, reduced_rank=True) -> bool:
        """
        Determine whether the encoded contrasts span the intercept.

        Args:
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether the contrast encoding used had reduced rank.
        """
        return not reduced_rank

    def get_drop_field(self, levels, reduced_rank=True) -> Union[int, str]:
        """
        Determine which column to drop to be full rank after this encoding.
        If this contrast encoding is already reduced in rank, then this method
        should return `None`.

        Args:
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether the contrast encoding used had reduced rank.
        """
        if reduced_rank:
            return None
        return self.get_coding_column_names(levels, reduced_rank=reduced_rank)[0]

    def get_factor_format(self, levels, reduced_rank=True):
        """
        The format to use when assigning feature names to each encoded feature.
        Formats can use two named substitutions: `name` and `field`; for
        example: "{name}[{field}]".

        Args:
            levels: The names of the levels/categories in the data.
            reduced_rank: Whether the contrast encoding used had reduced rank.
        """
        return self.FACTOR_FORMAT


class TreatmentContrasts(Contrasts):
    """
    Treatment (aka. dummy) coding.

    This contrast leads to comparisons of the mean of the dependent variable for
    each level with some reference level. If not specified, the reference level
    is taken to be the first level.
    """

    FACTOR_FORMAT = "{name}[T.{field}]"

    MISSING = object()

    def __init__(self, base=MISSING):
        self.base = base

    @Contrasts.override
    def _apply(self, dummies, levels, reduced_rank=True, sparse=False):
        if reduced_rank:
            drop_index = self._find_base_index(levels)
            mask = numpy.ones(len(levels), dtype=bool)
            mask[drop_index] = False
            return (
                dummies
                if sparse or isinstance(dummies, numpy.ndarray)
                else dummies.iloc
            )[:, mask]
        return dummies

    def _find_base_index(self, levels):
        if self.base is self.MISSING:
            return 0
        try:
            return levels.index(self.base)
        except ValueError as e:
            raise ValueError(
                f"Value `{repr(self.base)}` for `TreatmentContrasts.base` is not among the provided levels."
            ) from e

    @Contrasts.override
    def _get_coding_matrix(self, levels, reduced_rank=True, sparse=False):
        n = len(levels)
        if sparse:
            matrix = spsparse.eye(n).tocsc()
        else:
            matrix = numpy.eye(n)
        if reduced_rank:
            drop_level = self._find_base_index(levels)
            matrix = matrix[:, [i for i in range(matrix.shape[1]) if i != drop_level]]
        return matrix

    @Contrasts.override
    def get_coding_column_names(self, levels, reduced_rank=True):
        base_index = self._find_base_index(levels)
        if reduced_rank:
            return [level for i, level in enumerate(levels) if i != base_index]
        return levels

    @Contrasts.override
    def get_coefficient_row_names(self, levels, reduced_rank=True):
        base = levels[self._find_base_index(levels)]
        if reduced_rank:
            return [base, *(f"{level}-{base}" for level in levels if level != base)]
        return levels

    @Contrasts.override
    def get_drop_field(self, levels, reduced_rank=True) -> Union[int, str]:
        if reduced_rank:
            return None
        return self.base if self.base is not self.MISSING else levels[0]


class SASContrasts(TreatmentContrasts):
    """
    SAS (treatment) contrast coding.

    This constrasts generated by this class are the same as
    `TreatmentContrasts`, but with the reference level defaulting to the last
    level (the default in SAS).
    """

    @TreatmentContrasts.override
    def _find_base_index(self, levels):
        if self.base is self.MISSING:
            return len(levels) - 1
        try:
            return levels.index(self.base)
        except ValueError as e:
            raise ValueError(
                f"Value `{repr(self.base)}` for `SASContrasts.base` is not among the provided levels."
            ) from e

    @TreatmentContrasts.override
    def get_drop_field(self, levels, reduced_rank=True) -> Union[int, str]:
        if reduced_rank:
            return None
        return self.base if self.base is not self.MISSING else levels[-1]


class SumContrasts(Contrasts):
    """
    Sum (or Deviation) coding.

    These contrasts compare the mean of the dependent variable for each level
    (except the last, which is redundant) to the global average of all levels.
    """

    FACTOR_FORMAT = "{name}[S.{field}]"

    @Contrasts.override
    def _get_coding_matrix(self, levels, reduced_rank=True, sparse=False):
        n = len(levels)
        if not reduced_rank:
            return spsparse.eye(n).tocsc() if sparse else numpy.eye(n)
        contr = spsparse.eye(n, n - 1).tolil() if sparse else numpy.eye(n, n - 1)
        contr[-1, :] = -1
        return contr.tocsc() if sparse else contr

    @Contrasts.override
    def get_coding_column_names(self, levels, reduced_rank=True):
        if reduced_rank:
            return levels[:-1]
        return levels

    @Contrasts.override
    def get_coefficient_row_names(self, levels, reduced_rank=True):
        if reduced_rank:
            return ["avg", *(f"{level} - avg" for level in levels[:-1])]
        return levels


class HelmertContrasts(Contrasts):
    """
    (Reverse) Helmert coding.

    These contrasts compare the mean of the dependent variable for each
    successive level to the average all previous levels.
    """

    FACTOR_FORMAT = "{name}[H.{field}]"

    @Contrasts.override
    def _get_coding_matrix(self, levels, reduced_rank=True, sparse=False):
        n = len(levels)
        if not reduced_rank:
            return spsparse.eye(n).tocsc() if sparse else numpy.eye(n)

        contr = spsparse.lil_matrix((n, n - 1)) if sparse else numpy.zeros((n, n - 1))
        for i in range(len(levels) - 1):
            contr[i + 1, i] = i + 1
        contr[numpy.triu_indices(n - 1)] = -1

        return contr

    @Contrasts.override
    def get_coding_column_names(self, levels, reduced_rank=True):
        if reduced_rank:
            return levels[:-1]
        return levels

    @Contrasts.override
    def get_coefficient_row_names(self, levels, reduced_rank=True):
        if reduced_rank:
            return ["avg", *(f"{level} - rolling_avg" for level in levels[1:])]
        return levels


class DiffContrasts(Contrasts):
    """
    (Backward) Difference coding.

    These contrasts compare the mean of the dependent variable for each level
    with that of the previous level.
    """

    FACTOR_FORMAT = "{name}[D.{field}]"

    @Contrasts.override
    def _get_coding_matrix(self, levels, reduced_rank=True, sparse=False):
        n = len(levels)
        if not reduced_rank:
            return spsparse.eye(n).tocsc() if sparse else numpy.eye(n)
        contr = numpy.repeat([numpy.arange(1, n)], n, axis=0) / n
        contr[numpy.triu_indices(n, m=n - 1)] -= 1
        if sparse:
            return spsparse.csc_matrix(contr)
        return contr

    @Contrasts.override
    def get_coding_column_names(self, levels, reduced_rank=True):
        if reduced_rank:
            return levels[1:]
        return levels

    @Contrasts.override
    def get_coefficient_row_names(self, levels, reduced_rank=True):
        if reduced_rank:
            return [
                "avg",
                *(f"{level} - {ref}" for level, ref in zip(levels[1:], levels)),
            ]
        return levels


class PolyContrasts(Contrasts):
    """
    (Orthogonal) Polynomial coding.

    These "contrasts" represent a categorical variable that is assumed to have
    equal (or known) spacing/scores, and allow us to model non-linear polynomial
    behaviour of the dependent variable with respect to the ordered levels.
    """

    FACTOR_FORMAT = "{name}{field}"
    NAME_ALIASES = {
        1: ".L",
        2: ".Q",
        3: ".C",
    }

    def __init__(self, scores=None):
        self.scores = scores

    @Contrasts.override
    def _get_coding_matrix(self, levels, reduced_rank=True, sparse=False):
        n = len(levels)
        if not reduced_rank:
            return spsparse.eye(n).tocsc() if sparse else numpy.eye(n)
        scores = self.scores or numpy.arange(n)
        coding_matrix = poly(scores, degree=n - 1)
        if sparse:
            return spsparse.csc_matrix(coding_matrix)
        return coding_matrix

    @Contrasts.override
    def get_coding_column_names(self, levels, reduced_rank=True):
        if reduced_rank:
            return [
                self.NAME_ALIASES[d] if d in self.NAME_ALIASES else f"^{d}"
                for d in range(1, len(levels))
            ]
        return levels

    @Contrasts.override
    def get_coefficient_row_names(self, levels, reduced_rank=True):
        if reduced_rank:
            return ["avg", *self.get_coding_column_names(levels, reduced_rank=True)]
        return levels


class CustomContrasts(Contrasts):
    """
    Handle the custom contrast case when users pass in hand-coded contrast
    matrices.
    """

    def __init__(self, contrasts, names=None):
        if isinstance(contrasts, dict):
            if names is None:
                names = list(contrasts)
            contrasts = numpy.array([*contrasts.values()]).T
        else:
            contrasts = numpy.array(contrasts)

        if names is not None and len(names) != contrasts.shape[1]:
            raise ValueError(
                "Names must be aligned with the columns of the contrast array."
            )

        self.contrasts = contrasts
        self.contrast_names = names

    @Contrasts.override
    def _get_coding_matrix(self, levels, reduced_rank=True, sparse=False):
        if sparse:
            return spsparse.csc_matrix(self.contrasts)
        return self.contrasts

    @Contrasts.override
    def get_coding_column_names(self, levels, reduced_rank=True):
        if self.contrast_names:
            return self.contrast_names
        return list(range(1, self.contrasts.shape[1] + 1))

    @Contrasts.override
    def get_coefficient_row_names(self, levels, reduced_rank=True):
        return list(range(1, len(levels) + (0 if not reduced_rank else 1)))

    @Contrasts.override
    def get_spans_intercept(self, levels, reduced_rank=True) -> bool:
        return False

    @Contrasts.override
    def get_drop_field(self, levels, reduced_rank=True) -> Union[int, str]:
        return None


class ContrastsRegistry(type):
    """
    The contrast registry, which is exposed in formulae as "contr".
    """

    # Same as R
    helmert = HelmertContrasts
    poly = PolyContrasts
    sum = SumContrasts
    treatment = TreatmentContrasts
    SAS = SASContrasts

    # Extra
    diff = DiffContrasts
    custom = CustomContrasts
