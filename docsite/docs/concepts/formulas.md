This section introduces the basic notions and origins of formulas. If you are
already familiar with formulas from another context, you might want to skip
forward to the [Formula Grammer](grammar.md) or [User Guides](../guides/).

## Origins

Formulas were originally proposed by Wilkinson et al.[^1] to aid in the
description of ANOVA problems, but were popularised by the S language (and then
[R](http://search.r-project.org/R/library/stats/html/formula.html), as an
implementation of S) in the context of linear regression. Since then they have
been [extended in
R](https://cran.r-project.org/web/packages/Formula/Formula.pdf), and implemented
in Python (by [patsy](https://github.com/pydata/patsy)), in
[MATLAB](https://www.mathworks.com/help/stats/wilkinson-notation.html), in
[Julia](https://juliadata.github.io/DataFrames.jl/v0.9/man/formulas/), and quite
conceivably elsewhere. Each implementation has its own nuances and grammatical
extensions, including Formulaic's which are described more completely in the
[Formula Grammar](grammar.md) section of this manual.


## Why are they useful?

Formulas are useful because they provide a concise and explicit specification
for how data should be prepared for a model. Typically, the raw input data for a
model is stored in a dataframe, but the actual implementations of various
statistical methodologies (e.g. linear regression solvers) act on
two-dimensional numerical matrices that go by [several
names](https://en.wikipedia.org/wiki/Design_matrix) depending on the prevailing
nomenclature of your field, including "model matrices", "design matrices" and
"regressor matrices" (within Formulaic, we refer to them as "model matrices"). A
formula provides the necessary information required to automate much of the
translation of a dataframe into a model matrix suitable for ingestion into a
statistical model.

Suppose, for example, that you have a dataframe with \\(N\\) rows and three
numerical columns labelled: `y`, `a` and `b`. You would like to construct a
linear regression model for `y` based on `a`, `b` and their interaction: \\[ y =
\alpha + \beta_a a + \beta_b b + \beta_{ab} ab + \varepsilon \\] with
\\(\varepsilon \sim \mathcal{N}(0, \sigma^2)\\). Rather than manually
constructing the required matrices to pass to the regression solver, you could
specify a formula of form:
```
y ~ a + b + a:b
```
When furnished with this formula and the dataframe, Formulaic (or indeed any
other formula implementation) would generate two model matrix objects: an \\( N
\times 1 \\) matrix \\(Y\\) for the response variable `y`, and an \\( N \times 4
\\) matrix \\(X\\) for the input columns `intercept`, `a`, `b`, and `a * b`. You
can then directly pass these matrices to your regression solver, which
internally will solve for \\(\beta\\) in: \\[ Y = X\beta + \varepsilon. \\]

The true value of formulas becomes more apparent as model complexity increases,
where they can be a huge time-saver. For example:
```
~ (f1 + f2 + f3) * (x1 + x2 + scale(x3))
```
tells the formula interpreter to consider 16 fields of input data, corresponding
to an intercept (1), each of the `f*` fields (3), each of the `x*` fields (3),
and the combination of each `f` with each `x` (9). It also instructs the
materializer to ensure that the `x3` column is rescaled during the model matrix
materialization phase such that it has mean zero and standard error of 1. If any
of these columns is categorical in nature, they would by default also be
one-hot/dummy encoded. Depending on the formula interpreter (including
Formulaic), extra steps would also be taken to ensure that the resulting model
matrix is structurally full-rank.

As an added bonus, some formula implementations (including Formulaic) can
remember any choices made during the materialization process, and apply them to
consistently to new data, making it possible to easily generate new data that
conforms to the same structure as the training data. For example, the
`scale(...)` transform in the example above makes use of the mean and variance
of the column to be scaled. Any future data should, however, should not undergo
scaling based on its own mean and variance, but rather on the mean and variance
that was measured for the training data set (otherwise the new dataset will not
be consistent with the expectations of the trained model which will be
interpreting it).

## Limitations

Formulas are a very flexible tool, and can be augmented with arbitrary
user-defined transforms. However, some transformations required by certain
models may be more elegantly defined via a pre-formula dataframe operation or
post-formula model matrix operation. Another consideration is that the default
encoding and materialization choices for data are aligned with linear
regression. If you are using a tree model, for example, you may not be
interested in dummy encoding of "categorical" features, and this type of
transform would have to be explicitly noted in the formula. Nevertheless, even
in these cases, formulas are an excellent tool, and can often be used to greatly
simplify data preparation workflows.

## Where to from here?

To learn about the full set of features supported by the formula language as
implemented by Formulaic, please review the [Formula Grammar](grammar.md). To
get a feel for how you can use `formulaic` to transform your dataframes into
model matrices, please review the [Quickstart](../guides/quickstart.md).  For
more advanced use-cases, such as overriding or customising the implementations
of formula parsing, please refer to the [Advanced Usage](../guides/advanced.md)
section.

[^1]: Wilkinson, G. N., and C. E. Rogers. Symbolic description of factorial models for analysis of variance. J. Royal Statistics Society 22, pp. 392–399, 1973.
