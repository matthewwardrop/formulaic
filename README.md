# <img src="https://raw.githubusercontent.com/matthewwardrop/formulaic/main/docsite/docs/assets/images/logo_with_text.png" alt="Formulaic" height=100/>

[![PyPI - Version](https://img.shields.io/pypi/v/formulaic.svg)](https://pypi.org/project/formulaic/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/formulaic.svg)
![PyPI - Status](https://img.shields.io/pypi/status/formulaic.svg)
[![build](https://img.shields.io/github/workflow/status/matthewwardrop/formulaic/Run%20Tox%20Tests)](https://github.com/matthewwardrop/formulaic/actions?query=workflow%3A%22Run+Tox+Tests%22)
[![codecov](https://codecov.io/gh/matthewwardrop/formulaic/branch/main/graph/badge.svg)](https://codecov.io/gh/matthewwardrop/formulaic)
[![Code Style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

Formulaic is a high-performance implementation of Wilkinson formulas for Python.

**Note:** This project, while largely complete, is still a work in progress, and the API is subject to change between major versions (0.&lt;major&gt;.&lt;minor&gt;).

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
    - `pyarrow.Table`
  - output:
    - `pandas.DataFrame`
    - `numpy.ndarray`
    - `scipy.sparse.CSCMatrix`
- support for symbolic differentiation of formulas (and hence model matrices).

## Example code

```
import pandas
from formulaic import Formula

df = pandas.DataFrame({
    'y': [0,1,2],
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

## Benchmarks

Formulaic typically outperforms R for both dense and sparse model matrices, and vastly outperforms `patsy` (the existing implementation for Python) for dense matrices (`patsy` does not support sparse model matrix output).

![Benchmarks](https://github.com/matthewwardrop/formulaic/raw/main/benchmarks/benchmarks.png)

For more details, see [here](benchmarks/README.md).

## Related projects and prior art

- [Patsy](https://github.com/pydata/patsy): a prior implementation of Wilkinson formulas for Python, which is widely used (e.g. in statsmodels). It has fantastic documentation (which helped bootstrap this project), and a rich array of features.
- [StatsModels.jl `@formula`](https://juliastats.org/StatsModels.jl/stable/formula/): The implementation of Wilkinson formulas for Julia.
- [R Formulas](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/formula): The implementation of Wilkinson formulas for R, which is thoroughly introduced [here](https://cran.r-project.org/web/packages/Formula/vignettes/Formula.pdf). [R itself is an implementation of [S](https://en.wikipedia.org/wiki/S_%28programming_language%29), in which formulas were first made popular].
- The work that started it all: Wilkinson, G. N., and C. E. Rogers. Symbolic description of factorial models for analysis of variance. J. Royal Statistics Society 22, pp. 392–399, 1973.
