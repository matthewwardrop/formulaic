# <img src="https://raw.githubusercontent.com/matthewwardrop/formulaic/main/docsite/docs/assets/images/logo_with_text.png" alt="Formulaic" height=100/>

[![PyPI - Version](https://img.shields.io/pypi/v/formulaic.svg)](https://pypi.org/project/formulaic/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/formulaic.svg)
![PyPI - Status](https://img.shields.io/pypi/status/formulaic.svg)
[![build](https://img.shields.io/github/actions/workflow/status/matthewwardrop/formulaic/tests.yml?branch=main)](https://github.com/matthewwardrop/formulaic/actions?query=workflow%3A%22Run+Tox+Tests%22)
[![docs](https://img.shields.io/github/actions/workflow/status/matthewwardrop/formulaic/publish_docs.yml?label=docs)](https://matthewwardrop.github.io/formulaic/)
[![codecov](https://codecov.io/gh/matthewwardrop/formulaic/branch/main/graph/badge.svg)](https://codecov.io/gh/matthewwardrop/formulaic)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

Formulaic is a high-performance implementation of Wilkinson formulas for Python.

- **Documentation**: https://matthewwardrop.github.io/formulaic
- **Source Code**: https://github.com/matthewwardrop/formulaic
- **Issue tracker**: https://github.com/matthewwardrop/formulaic/issues


It provides:

- high-performance dataframe to model-matrix conversions.
- support for reusing the encoding choices made during conversion of one data-set on other datasets.
- extensible formula parsing.
- extensible data input/output plugins, with implementations for:
  - input:
    - `pandas.DataFrame`
    - Any dataframe representation supported by [`narwhals`](https://narwhals-dev.github.io/narwhals/) including
      - `pyarrow.Table`
      - `polars.DataFrame`
      - ...
  - output:
    - `pandas.DataFrame`
    - `numpy.ndarray`
    - `scipy.sparse.CSCMatrix`
    - `narwhals` dataframe passthrough when using narwhals dataframes.
- support for symbolic differentiation of formulas (and hence model matrices).
- and much more.

## Example code

```
import pandas
from formulaic import Formula

df = pandas.DataFrame({
    'y': [0, 1, 2],
    'x': ['A', 'B', 'C'],
    'z': [0.3, 0.1, 0.2],
})

y, X = Formula('y ~ x + z').get_model_matrix(df)
```

`y = `
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
  </tbody>
</table>

`X = `
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Intercept</th>
      <th>x[T.B]</th>
      <th>x[T.C]</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>

Note that the above can be short-handed to:

```
from formulaic import model_matrix
model_matrix('y ~ x + z', df)
```

## Benchmarks

Formulaic typically outperforms R for both dense and sparse model matrices, and vastly outperforms `patsy` (the existing implementation for Python) for dense matrices (`patsy` does not support sparse model matrix output).

![Benchmarks](https://github.com/matthewwardrop/formulaic/raw/main/benchmarks/benchmarks.png)

For more details, see [here](benchmarks/README.md).

## Related projects and prior art

- [Patsy](https://github.com/pydata/patsy): a prior implementation of Wilkinson formulas for Python, which is widely used (e.g. in statsmodels). It has fantastic documentation (which helped bootstrap this project), and a rich array of features.
- [StatsModels.jl `@formula`](https://juliastats.org/StatsModels.jl/stable/formula/): The implementation of Wilkinson formulas for Julia.
- [R Formulas](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/formula): The implementation of Wilkinson formulas for R, which is thoroughly introduced [here](https://cran.r-project.org/web/packages/Formula/vignettes/Formula.pdf). [R itself is an implementation of [S](https://en.wikipedia.org/wiki/S_%28programming_language%29), in which formulas were first made popular].
- The work that started it all: Wilkinson, G. N., and C. E. Rogers. Symbolic description of factorial models for analysis of variance. J. Royal Statistics Society 22, pp. 392–399, 1973.

## Used by

Below are some of the projects that use Formulaic:

- [Glum](https://github.com/Quantco/glum): High performance Python GLM's with all the features.
- [Lifelines](https://github.com/camDavidsonPilon/lifelines): Survival analysis in Python.
- [Linearmodels](https://github.com/bashtage/linearmodels): Additional linear models including instrumental variable and panel data models that are missing from statsmodels.
- [Pyfixest](https://github.com/s3alfisc/pyfixest): Fast High-Dimensional Fixed Effects Regression in Python following fixest-syntax.
- [Tabmat](https://github.com/Quantco/tabmat): Efficient matrix representations for working with tabular data.
- Add your project here!
