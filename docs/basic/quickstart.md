This document provides high-level documentation on how to get started using
Formulaic. For deeper documentation about the internals, please refer to the
[Advanced Usage](../advanced/intro.md) documentation.

## What is a "formula"?

The purpose of a "formula" is to represent the model you would like to build in
a way that can be programmatically parsed. You can then use the parsed
representation to build the dataset (called "model matrices") you need for your
model.

For example, suppose you would like to create a linear regression model for `y`
based on `a`, `b` and their interaction:
\\[ y = \alpha + \beta_a a + \beta_b b + \beta_{ab} ab + \varepsilon \\]
with \\(\varepsilon\\) being mean zero normally distributed random variable.
Rather than manually constructing the required matrices to pass to the
regression solver, you could just specify a formula of form:
```
y ~ a + b + a:b
```
If you provide this formula to formulaic (or any other formula implementation)
along with a dataframe of length \\( N \\) with fields for `y`, `a` and `b`, you
will receive two matrix objects: a \\( N \times 1 \\) matrix \\(Y\\) for `y`,
and a \\( N \times 4 \\) matrix \\(X\\)  for the `intercept`, `a`, `b`, and `a *
b` columns. You can then directly pass these matrices to your regression solver,
which internally will solve for \\(\beta\\) in:
\\[ Y = X\beta + \varepsilon. \\]

For more on how to specify formulas, please refer to the
[Formula Grammar](formulas.md).


## Building Model Matrices

In formulaic, the simplest way to build your model matrices is to use the
high-level `model_matrix` function:

```
import pandas
from formulaic import model_matrix

df = pandas.DataFrame({
    'y': [0, 1, 2],
    'a': ['A', 'B', 'C'],
    'b': [0.3, 0.1, 0.2],
})

y, X = model_matrix("y ~ a + b + a:b", df)
# This is short-hand for:
# y, X = formulaic.Formula('y ~ a + b + a:b').get_model_matrix(df)
# This lower-level API discussed in the Advanced Usage documentation.
```

`y =`
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
      <th>a[T.B]</th>
      <th>a[T.C]</th>
      <th>b</th>
      <th>a[T.B]:b</th>
      <th>a[T.C]:b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>

You will notice that the categorical values for `a` have been one-hot encoded,
and to ensure structural full-rankness of `X`[^1], one level has been dropped
from `a`. For more details about how this guarantees that the matrix is full-rank,
please refer to the excellent [patsy documentation](https://patsy.readthedocs.io/en/latest/formulas.html).
If you are not using the model matrices for regression, and don't care if the
matrix is not full-rank, you can pass `ensure_full_rank=False`:

```
X = model_matrix("a + b + a:b", df, ensure_full_rank=False)
```

`X =`
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Intercept</th>
      <th>a[T.A]</th>
      <th>a[T.B]</th>
      <th>a[T.C]</th>
      <th>b</th>
      <th>a[T.A]:b</th>
      <th>a[T.B]:b</th>
      <th>a[T.C]:b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>

Note that the dropped level in `a` has been restored.

### Sparse Model Matrices

By default, the generated model matrices are dense. In some case, particularly
in large datasets with many categorical features, dense model matrices become
hugely memory inefficient (since most entries of the data will be zero).
Formulaic allows you to directly generate sparse model matrices using:
```
X = model_matrix("a + b + a:b", df, sparse=True)
```
In this example, `X` is a \\( 6 \times 3 \\) `scipy.sparse.csc_matrix` instance.

[^1]: `X` must be full-rank in order for the regression algorithm to invert a matrix derived from `X`.
