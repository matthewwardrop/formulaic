The sole focus of `formulaic` is to allow users to use formulas (see below) to
help them transform dataframes into a form suitable for ingestion into various
modelling frameworks.

Formulas were originally proposed by Wilkinson et al.[^1] to aid in the
description of ANOVA problems, but were popularised by the S language (and then
R, as an implementation of S) in the context of linear regression. Since then
they have been implemented in Python (by
[patsy](https://github.com/pydata/patsy)), in
[MATLAB](https://www.mathworks.com/help/stats/wilkinson-notation.html), in
[Julia](https://juliadata.github.io/DataFrames.jl/v0.9/man/formulas/), and quite
conceivably elsewhere. Each implementation has its own nuances and grammatical
extensions, including Formulaic's which are described in the
[Formula Grammar](formulas.md).

To get a feel for how you can use `formulaic` to transform your dataframes into
please review the [Quick Start](quickstart.md). To learn about the full set of
features supported by the formula language as implemented by Formulaic, please
review the [Formula Grammar](formulas.md). For more advanced use-cases, such as
overriding or customising the implementations of formula parsing, please refer
to the [Advanced Usage](../advanced/intro.md) section.

[^1]: Wilkinson, G. N., and C. E. Rogers. Symbolic description of factorial models for analysis of variance. J. Royal Statistics Society 22, pp. 392–399, 1973.
