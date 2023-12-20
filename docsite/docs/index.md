<img src="assets/images/logo_with_text.png" style="max-width: 600px">

Formulaic is a high-performance implementation of Wilkinson formulas for Python,
which are very useful for transforming dataframes into a form suitable for
ingestion into various modelling frameworks (especially linear regression).

- **Source Code**: [https://github.com/matthewwardrop/formulaic](https://github.com/matthewwardrop/formulaic)
- **Issue tracker**: [https://github.com/matthewwardrop/formulaic/issues](https://github.com/matthewwardrop/formulaic/issues)

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

with more to come!
