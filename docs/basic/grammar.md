This section of the documentation describes the formula grammar used by
Formulaic. It is almost identical that used by patsy and R, and so most formulas
should work without modification. However, there are some differences, which are
called out below.


## Operators

In this section, we introduce a complete list of the _grammatical_ operators
that you can use by default in your formulas. They are listed such that each
section (demarcated by "-----") has higher precedence then the block that
follows. When you write a formula involving several operators of different
precedence, those with higher precedence will be resolved first. "Arity" is the
number of arguments the operator takes. Within operators of the same precedence,
all binary operators are evaluated from left to right (they are
left-associative). To highlight differences in grammar betweeh formulaic, patsy
and R, we highlight any differences below. If there is a checkmark the
Formulaic, Patsy and R columns, then the grammar is consistent across all three,
unless otherwise indicated.

| Operator | Arity | Description | Formulaic | Patsy | R |
|---------:|:-----:|:------------|:---------:|:-----:|:-:|
| `"..."`[^1] | 1 | String literal. | ✓ | ✓ | 🗙 |
| `[0-9]+\.[0-9]+`[^1] | 1 | Numerical literal. | ✓ | 🗙 | 🗙 |
| `` `...` ``[^1] | 1 | Quotes fieldnames within the incoming dataframe, allowing the use of special characters, e.g. `` `my|special$column!` `` | ✓ | 🗙 | ✓ |
| `{...}`[^1] | 1 | Quotes python operations, as a more convenient way to do Python operations than `I(...)`, e.g. `` {`my|col`**2} `` | ✓ | 🗙 | 🗙 |
| `<function>(...)`[^1] | 1 | Python transform on column, e.g. `my_func(x)` which is equivalent to `{my_func(x)}` | ✓[^2] | ✓ | 🗙 |
|-----|
| `(...)` | 1 | Groups operations, overriding normal precedence rules. All operations with the parentheses are performed before the result of these operations is permitted to be operated upon by its peers. | ✓ | ✓ | ✓ |
|-----|
| ** | 2 | Includes all n-th order interactions of the terms in the left operand, where n is the (integral) value of the right operand, e.g. `(a+b+c)**2` is equivalent to `a + b + c + a:b + a:c + b:c | ✓ | ✓ | ✓ |
| ^ | 2 | Alias for `**`. | 🗙 | 🗙[^3] | ✓ |
|-----|
| `:` | 2 | Adds a new term that corresponds to the interaction of its operands (i.e. their elementwise product). | ✓ | ✓ | ✓ |
|-----|
| `*` | 2 | Includes terms for each of the additive and interactive effects of the left and right operands, e.g. `a * b` is equivalent to `a + b + a:b`. | ✓ | ✓ | ✓ |
| `/` | 2 | Adds terms describing nested effects. It expands to the addition of a new term for the left operand and the interaction of all left operand terms with the right operand, i.e `a / b` is equivalent to `a + a:b`, `(a + b) / c` is equivalent to `a + b + a:b:c`, and `a/(b+c)` is equivalent to `a + a:b + a:c`.[^4] | ✓ | ✓ | ✓ |
| `%in%` | 2 | Alias for `/`. | 🗙 | 🗙 | ✓ |
|-----|
| `+` | 2 | Adds a new term to the set of features. | ✓ | ✓ | ✓ |
| `-` | 2 | Removes a term from the set of features (if present). | ✓ | ✓ | ✓ |
| `+` | 1 | Returns the current term unmodified (not very useful). | ✓ | ✓ | ✓ |
| `-` | 1 | Negates a term (only implemented for 0, in which case it is replaced with `1`). | ✓ | ✓ | ✓ |
|-----|
| `~` | 1,2 | Separates the target features from the input features. If absent, it is assumed that we are considering only the the input features. Unless otherwise indicated, it is assumed that the input features implicitly include an intercept. | ✓[^5] | ✓ | ✓ |


## Transforms

Formulaic supports arbitrary transforms, any of which can also preserve state so
that new data can undergo the same transformation as that used during modelling.
The currently implemented transforms are shown below. Commonly used transforms
that have *not* been implemented by `formualaic` are explicitly noted also.

| Transform | Description | Formulaic | Patsy | R |
|----------:|:------------|:---------:|:-----:|:-:|
| `I(...)` | Identity transform, allowing arbitrary Python/R operations, e.g. `I(x+y)`. Note that in `formulaic`, it is more idiomatic to use `{x+y}`. | ✓ | ✓ | ✓ |
| `C(...)` | Categorically encode a column, e.g. `C(x)` | partial[^6] | ✓ | ✓ |
| `center(...)` | Shift column data so mean is zero. | ✓ | ✓ | 🗙 |
| `scale(...)` | Shift column so mean is zero and variance is 1. | ✓ | ✓[^7] | ✓ |
| `standardize(...)` | Alias of `scale`. | 🗙 | ✓ | 🗙 |
| `bs(...)` | Generates a B-Spline basis, allowing non-linear fits. | ✓ | ✓ | ✓ |
| `cr(...)` | Generates a natural cubic spline basis, allowing non-linear fits. | 🗙 | ✓ | ✓ |
| `cc(...)` | Generates a cyclic cubic spline basis, allowing non-linear fits. | 🗙 | ✓ | ✓ |
| `te(...)` | Generates a tensor product smooth. | 🗙 | ✓ | ✓ |
| ...       | Others? Contributions welcome!     | ? | ? | ? |

!!! note
    Formulaic does not (yet) support including extra terms in the formula that
    will not result in additions to the dataframe, for example model annotations
    like R's `offset(...)`.

## Behaviours and Conventions

Beyond the formula operator grammar itself there are some differing behaviours
and conventions of which you should be aware.

  - Formulaic follows Patsy and then enhanced `Formula` R package in that both
    sides of the `~` operator are treated considered to be using the formula
    grammar, with the only difference being that the right hand side attracts an
    intercept by default. In vanilla R, the left hand side is treated as R code
    (and so `x + y ~ z` would result in a single column on the left-hand-side).
    You can recover vanilla R's behaviour by nesting the operations in a Python
    operator block (as described in the operator table): `{y1 + y2} ~ a + b`.
  - Formula terms in Formulaic are always sorted first by the order of the
    interaction, and then alphabetically. In R and patsy, this second ordering
    is done in the order that columns were introduced to the formula (patsy
    additionally sorts by which fields are involved in the interactions). As a
    result formulas generated by `formulaic` with the same set of fields will
    always generate the same model matrix.
  - Formulaic follows patsy's more rigourous handling of whether or not to
    include an intercept term. In R, `b-1` and `(b-1)` both do not have an
    intercept, whereas in Formulaic and Patsy the parentheses are resolved
    first, and so the first does not have an intercept and the second does
    (because and implicit '1 +' is added prepended to the right hand side of the
    formula).
  - Formulaic borrows a clever algorithm introduced by Patsy to carefully choose
    where to reduce the rank of the model matrix in order to ensure that the
    matrix is structurally full rank. This avoids producing over-specified model
    matrices in contexts that R would (since it only considers local full-rank
    structure, rather than global structure). You can read more about this in
    [Patsy's documentation](https://patsy.readthedocs.io/en/latest/formulas.html).


[^1]: This "operator" is actually part of the tokenisation process.
[^2]: Formulaic additionally supports quoted fields with special characters, e.g. `` my_func(`my|special+column`) ``.
[^3]: The caret operator is not supported, but will not cause an error. It is ignored by the patsy formula parser, and treated as XOR Python operation on column.
[^4]: This somewhat confusing operator is useful when you want to include hierachical features in your data, and where certain interaction terms do not make sense (particularly in ANOVA contexts). For example, if `a` represents countries, and `b` represents cities, then the full product of terms from `a * b === a + b + a:b` does not make sense, because any value of `b` is guaranteed to coincide with a value in `a`, and does not independently add value. Thus, the operation `a / b === a + a:b` results in more sensible dataset. As a result, the `/` operator is right-distributive, since if `b` and `c` were both nested in `a`, you would want `a/(b+c) === a + a:b + a:c`. Likewise, the operator is not left-distributive, since if `c` is nested under both `a` and `b` separately, then you want `(a + b)/c === a + b + a:b:c`. Lastly, if `c` is nested in `b`, and `b` is nested in `a`, then you would want `a/b/c === a + a:(b/c) === a + a:b + a:b:c`.
[^5]: Formulaic additionally supports more than one `~` which splits the features into as many groups as you like.
[^6]: Formulaic only supports one-hot encoding, and does not yet support arbitrary contrast matrices or specification which field to leave out in reduced rank, etc.
[^7]: Patsy uses the `rescale` keyword rather than `scale`, but provides the same functionality.
