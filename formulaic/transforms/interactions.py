from typing import TYPE_CHECKING, Any

import narwhals.stable.v1 as narwhals

from formulaic.materializers.types import FactorValues
from formulaic.transforms import stateful_transform
from formulaic.transforms.contrasts import C

if TYPE_CHECKING:
    from formulaic.model_spec import ModelSpec  # pragma: no cover


@stateful_transform
def i(*args, _spec=None, _materializer=None):
    """The 'interaction' transform, which creates interaction terms between non-null
    combinations of the input arguments"""

    # TODO: Keep track of encoder state.

    if len(args) == 0:
        return {}

    if not _materializer:
        raise RuntimeError("The 'i' transform requires a materializer context")

    def encoder(
        values: dict[str, Any],
        reduced_rank: bool,
        drop_rows: list[int],
        encoder_state: dict[str, Any],
        model_spec: ModelSpec,
    ) -> FactorValues:
        required_terms = narwhals.DataFrame.from_dict(
            {
                arg.name: narwhals.from_native(
                    arg,
                    series_only=True,
                )
                for arg in args
            }
        ).unique().sort(list(values.keys()))

        encoded = {}
        categorical_factors = set()
        for name, arg in values.items():
            if isinstance(arg, FactorValues):
                if arg.__formulaic_metadata__.encoder:
                    encoded[name] = arg.__formulaic_metadata__.encoder(
                        values=arg, reduced_rank=False, drop_rows=drop_rows, encoder_state={}, model_spec=_spec
                    )
                    if not narwhals.dependencies.is_into_series(encoded[name]):
                        categorical_factors.add(name)
                else:
                    encoded[name] = arg
            elif _materializer._is_categorical(arg):
                categorical_factors.add(name)
                encoded[name] = dict(
                    C(arg).__formulaic_metadata__.encoder(
                        arg, reduced_rank=False, drop_rows=drop_rows, encoder_state={}, model_spec=_spec
                    )
                )
            else:
                encoded[name] = arg

        out = {}
        for row in required_terms.iter_rows(named=True):
            factors = []
            for name, value in row.items():
                if name in categorical_factors:
                    if value not in encoded[name]:
                        break
                    factors.append(
                        {
                            getattr(values[name], "format", "{name}[{field}]").format(name=name, field=value): encoded[name][value]
                        }
                    )
                else:
                    factors.append({name: encoded[name]})
            else:
                out.update(
                    _materializer._get_columns_for_term(
                        factors=factors,
                        spec=_spec,
                    )
                )

        return FactorValues(
            out,
            format="{field}",
            spans_intercept=False,
        )

    return FactorValues(
        {
            arg.name: arg
            for arg in args
        },
        kind="categorical",
        spans_intercept=False,
        encoder=encoder,
    )
