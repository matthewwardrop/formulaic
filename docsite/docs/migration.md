The default Formulaic parser and materialization configuration is designed to be
highly compatibly with existing Wilkinson formula implementations in R and
Python; however there are some differences which are highlighted here. If you
find other differences, feel free to submit a PR to update this documentation.

## Migrating from `patsy`

[Patsy](https://github.com/pydata/patsy) has been the go-to implementation of
Wilkinson formulae for Python use-cases for many years, and Formulaic should be
largely a drop-in replacement, while bringing order of magnitude improvements in
runtime performance and greater extensibility. Being written in the same
language (Python) there are two separate migration concerns: input/output and
API migrations, which will be explored separately below.

### Input/Output changes

The primary inputs to `patsy` are a formula string, and pandas dataframe from
which features referenced in the formula are drawn. The output is a model matrix
(called a design matrix in `patsy`). We focus here on any potentially breaking
behavioural differences here, rather than ways in which Formulaic extends the
functionality available in `patsy`.

* The `^` operator is interpreted as exponentiation, rather than Python's XOR
    binary operator.
* Contrast encoding is recommended to follow R-style conventions e.g.
    `C(x, contr.treatment)`. For greater compatibility with `patsy` we add to the
    transform namespace `Treatment`, `Poly`, `Sum`, `Helmert` and `Diff`,
    allowing formulae like `C(x, Poly)` or `C(x, Treatment(reference='x'))` to
    work as expected, with the following caveats:
    - The signature of `C` is `C(data, contrasts=None, *, levels=None)` as
        compared to `C(data, contrast=None, levels=None)` from `patsy`.
    - The `Sum` contrast does not offer an `omit` option to specify the index of
        the omitted column.
* Feature rescaling is recommended to follow R conventions e.g. `scale(x)`, but
    compatibility shims for `standardize(x)` are added for greater
    compatibility with `patsy`. Note that the `standardize` shim follows patsy
    argument kwarg naming conventions, but `scale` uses `scale` instead of
    `rescale`, following R.
* The order of the model matrix columns will differ by default. Patsy groups
    columns by the numerical features from which they derived, then sorts by
    interaction order, and then by the order in which features were added into
    the formula. Formulaic does not by default do the clustering by numerical
    factors. This behaviour can be restored by passing
    `cluster_by="numerical_factors"` to `model_matrix` or any of the
    `.get_model_matrix(...)` methods.
* Formulaic does not yet have implementations for natural and cyclic cubic basis
    splines (`cr` and `cc`) or tensor smoothing (`te`) stateful transforms.

### API translations

`patsy` offers two high-level user-facing entrypoints: `patsy.dmatrix` and
`patsy.dmatrices`, depending on whether you have both left- and right-hand sides
present. In `formulaic`, we offer a single entrypoint for both cases:
`model_matrix`.

In the vast majority of cases, a simple substitution of `dmatrix` or `dmatrices`
with `model_matrix` will achieve the desired the result; however there are some
differences in signature that could trip up a naive copy and replace. Patsy's
`dmatrix` signature is:
```
patsy.dmatrix(
    formula_like,
    data={},
    eval_env=0,
    NA_action='drop',
    return_type='matrix',
)
```
whereas `model_matrix` has a signature of:
```
formulaic.model_matrix(
    spec: FormulaSpec,  # accepts any formula-like spec (include model matrices and specs)
    data: Any,  # accepts any supported data structure (include pandas DataFrames)
    *,
    context: Union[int, Mapping[str, Any]] = 0,  # equivalent to `eval_env`
    **spec_overrides,  # Additional overrides for generated `ModelSpec`, including `na_action` and `output` (similar to `return_type`).
)
```

If you are integrating Formulaic into your library, it is highly recommended to
use the `Formula()` API directly rather than `model_matrix`, which by default
will add all variables in the local context into the evaluation environment
(just like `dmatrix`). This allows you to better isolate and control the
behaviour of the Formula parsing.


## Migrating from R

Most formulae that work in R will work without modification, including those
written against the enhanced [R Formula package](https://cran.r-project.org/web/packages/Formula/)
that supports multi-part formulae. However, there are a few caveats that are
worth calling out:

* As in the enhanced [R Formula package](https://cran.r-project.org/web/packages/Formula/),
    the left hand side of formulae can have multiple terms; the only difference
    between the left- and right-hand sides being that an intercept is
    automatically added on the right.
* Exponentiation will not work using the `^` operator within an `I` transform;
    e.g. `I(x^2)`. This is because this is treated as Python code, and so you
    should use `I(x**2)` or `{x**2}` instead.
* Intercept inclusion/exclusion directives are handled more rigorously,
    following the conventions of `patsy`. In particular, order of operations are
    respected when evaluating intercept directives, and so: `1 + (b - 1)` would
    result in the intercept remaining (since `(b-1)` would be evaluated first to
    `b`, resulting in `1 + b`), whereas in R the intercept would have been
    dropped.
* Model matrices are guaranteed to be structurally full-rank no matter how
    categorical variables are interacted, whereas R will sometimes become
    confused and output over- or under-specified model matrices. The algorithm
    used is the same as that found in `patsy`. Using capital letters to
    represent categorical variables, and lower-case letters to represent
    numerical ones, the difference from R will become apparent in two cases:
    1. When categories are interacted in the presence of intercept. e.g.:
        `1 + A:B`. In this case, R does not account for the fact that `A:B`
        spans the intercept, and so does not rank reduce the product, and thus
        generates an over-specified matrix. This affects higher-order
        interactions also.
    2. When categories are interacted with numerical features alongside
        interactions with categorical features. e.g.: `0 + A:x + B:C`. Here we
        use `0 +` to avoid the previous bug, but unfortunately when R is
        checking whether to reduce the rank of the categorical features during
        encoding, it assumes that all involved features are categorical, and
        thus unnecessarily reduces the rank of `C`, resulting in an
        under-specified matrix. This affects higher-order interactions also.
* Formulaic does not (yet) support including extra "metadata" terms in the
    formula that will not result in additions to the model matrix, for example
    model annotations like R's `offset(...)`.
* Some transforms that are commonly available in R may not be available.

For more details, refer to the [Formula Grammar](../guides/grammar/).
