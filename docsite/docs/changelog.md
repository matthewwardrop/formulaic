For changes since the latest tagged release, please refer to the
[git commit log](https://github.com/matthewwardrop/formulaic/commits/main).

---

## 0.4.0 (10 Aug 2022)

This is a major new release with some new features, greatly improved ergonomics
for structured formulae, matrices and specs, and a few small breaking changes
(most with backward compatibility shims). All users are encouraged to upgrade.

**Breaking changes:**

* `include_intercept` is no longer an argument to `FormulaParser.get_terms`;
  and is instead an argument of the `DefaultFormulaParser` constructor. If you
  want to modify the `include_intercept` behaviour, please use:
  ```python
  Formula("y ~ x", _parser=DefaultFormulaParser(include_intercept=False))
  ```
* Accessing terms via `Formula.terms` is deprecated since `Formula` became a
  subclass of `Structured[List[Terms]]`. You can directly iterate over, and/or
  access nested structure on the `Formula` instance itself. `Formula.terms`
  has a deprecated property which will return a reference to itself in order to
  support legacy use-cases. This will be removed in 1.0.0.
* `ModelSpec.feature_names` and `ModelSpec.feature_columns` are deprecated in
  favour of `ModelSpec.column_names` and `ModelSpec.column_indices`. Deprecated
  properties remain in-place to support legacy use-cases. These will be removed
  in 1.0.0.

**New features and enhancements:**

* Structured formulae (and their derived matrices and specs) are now mutable.
  Internally `Formula` has been refactored as a subclass of
  `Structured[List[Terms]]`, and can be incrementally built and modified. The
  matrix and spec outputs now have explicit subclasses of `Structured`
  (`ModelMatrices` and `ModelSpecs` respectively) to expose convenience methods
  that allow these objects to be largely used interchangeably with their
  singular counterparts.
* `ModelMatrices` and `ModelSpecs` arenow surfaced as top-level exports of the
  `formulaic` module.
* `Structured` (and its subclasses) gained improved integration of nested tuple
  structure, as well as support for flattened iteration, explicit mapping
  output types, and lots of cleanups.
* `ModelSpec` was made into a dataclass, and gained several new
  properties/methods to support better introspection and mutation of the model
  spec.
* `FormulaParser` was renamed `DefaultFormulaParser`, and made a subclass of the
  new formula parser interface `FormulaParser`. In this process
  `include_intercept` was removed from the API, and made an instance attribute
  of the default parser implementation.

**Bugfixes and cleanups:**

* Fixed AST evaluation for large formulae that caused the evaluation to hit the
  recursion limit.
* Fixed sparse categorical encoding when the dataframe index is not the standard
  range index.
* Fixed a bug in the linear constraints parser when more than two constraints
  were specified in a comma-separated string.
* Avoid implicit changing of the sparsity structure of CSC matrices.
* If manually constructed `ModelSpec`s are provided by the user during
  materialization, they are updated to reflect the output-type chosen by the
  user, as well as whether to ensure full rank/etc.
* Allowed use of older pandas versions. All versions >=1.0.0 are now supported.
* Various linting cleanups as `pylint` was added to the CI testing.

**Documentation:**

* Apart from the `.materializer` submodule, most code now has inline
  documentation and annotations.

---

## 0.3.4 (1 May 2022)

This is a backward compatible major release that adds several new features.

**New features and enhancements:**

* Added support for customizing the contrasts generated for categorical
  features, including treatment, sum, deviation, helmert and custom contrasts.
* Added support for the generation of linear constraints for `ModelMatrix`
  instances (see `ModelMatrix.model_spec.get_linear_constraints`).
* Added support for passing `ModelMatrix`, `ModelSpec` and other formula-like
  objects to the `model_matrix` sugar method so that pre-processed formulae can
  be used.
* Improved the way tokens are manipulated for the right-hand-side intercept and
  substitutions of `0` with `-1` to avoid substitutions in quoted contexts.

**Bugfixes and cleanups:**

* Fixed variable sanitization during evaluation, allowing variables with
  special characters to be used in Python transforms; for example:
  ```bs(`my|feature%is^cool`)```.
* Fixed the parsing of dictionaries and sets within python expressions in the
  formula; for example: `C(x, {"a": [1,2,3]})`.
* Bumped requirement on `astor` to >=0.8 to fix issues with ast-generation in
  Python 3.8+ when numerical constants are present in the parsed python
  expression (e.g. "bs(x, df=10)").

## 0.3.3 (4 April 2022)

This is a minor patch release that migrates the package tooling to
[poetry](https://python-poetry.org/); solving a version inconsistency when
packaging for `conda`.

## 0.3.2 (17 March 2022)

This is a minor patch release that fixes an attempt to import `numpy.typing`
when numpy is not version 1.20 or later.

## 0.3.1 (15 March 2022)

This is a minor patch release that fixes the maintaining of output types,
NA-handling, and assurance of full-rank for factors that evaluate to pre-encoded
columns when constructing a model matrix from a pre-defined ModelSpec. The
benchmarks were also updated.

## 0.3.0 (14 March 2022)

This is a major new release with many new features, and a few small breaking
changes. All users are encouraged to upgrade.

**Breaking changes:**

* The minimum supported version of Python is now 3.7 (up from 3.6).
* Moved transform implementations from `formulaic.materializers.transforms` to
    the top-level `formulaic.transforms` module, and ported all existing
    transforms to output `FactorValues` types rather than dictionaries.
    `FactorValues` is an object proxy that allows output types like
    `pandas.DataFrame`s to be used as they normally would, with some additional
    metadata for formulaic accessible via the `__formulaic_metadata__`
    attribute. This makes non-formula direct usage of these transforms much more
    pleasant.
* `~` is no longer a generic formula separator, and can only be used once in a
    formula. Please use the newly added `|` operator to separate a formula into
    multiple parts.

**New features and enhancements:**

* Added support for "structured" formulas, and updated the `~` operator to use
    them. Structured formulas can have named substructures, for example: `lhs`
    and `rhs` for the `~` operator. The representation of formulas has been
    updated to show this structure.
* Added support for context-sensitivity during the resolution of operators,
    allowing more flexible operators to be implemented (this is exploited by the
    `|` operator which splits formulas into multiple parts).
* The `formulaic.model_matrix` syntactic sugar function now accepts `ModelSpec`
    and `ModelMatrix` instances as the "formula" spec, making generation of
    matrices with the same form as previously generated matrices more
    convenient.
* Added the `poly` transform (compatible with R and patsy).
* `numpy` is now always available in formulas via `np`, allowing formulas like
    `np.sum(x)`. For convenience, `log`, `log10`, `log2`, `exp`, `exp10` and
    `exp2` are now exposed as transforms independent of user context.
* Pickleability is now guaranteed and tested via unit tests. Failure to pickle
    any formulaic metadata object (such as formulas, model specs, etc) is
    considered a bug.
* The capturing of user context for use in formula materialization has been
    split out into a utility method `formulaic.utils.context.capture_context()`.
    This can be used by libraries that wrap Formulaic to capture the variables
    and/or transforms available in a users' environment where appropriate.

**Bugfixes and cleanups:**

* Migrated all code to use the Black style.
* Increased unit testing coverage to 100%.
* Fixed mis-alignment in the right- and left-hand sides of formulas if there
    were nulls at different indices.
* Fixed basis spline transforms ignoring state, fixed generated splines for
    large numbers of knots, and fixed specification of knots via non-list
    datatypes.
* Fixed category order being inconsistent if categories are explicitly ordered
    differently in the underlying data.
* Lots of other minor nits and cleanups.

**Documentation:**

* The structure of the docsite has been improved (but is still incomplete).
* The `.parser` and `.utils` modules of Formulaic are now inline documented
    and annotated.

---

## 0.2.4 (9 July 2021)

This is a minor release that fixes an issue whereby the ModelSpec instances
attached to ModelMatrix objects would keep reference to the original data,
greatly inflating the size of the ModelSpec.

## 0.2.3 (4 February 2021)

This release is identical to v0.2.2, except that the source distribution now
includes the docs, license, and tox configuration.

## 0.2.2 (4 February 2021)

This is a minor release with one bugfix.

- Fix pandas model matrix outputs when constants are generated as part of model
  matrix construction and the incoming dataframe has a custom rather than range
  index.

## 0.2.1 (22 January 2021)

This is a minor patch release that brings in some valuable improvements.

- Keep track of the pandas dataframe index if outputting a pandas `DataFrame`.
- Fix using functions in formulae that are nested within a module or class.
- Avoid crashing when an attempt is made to generate an empty model matrix.
- Enriched setup.py with long description for a better experience on PyPI.

## 0.2.0 (21 January 2021)

This is major release that brings in a large number of improvements, with a huge
number of commits. Some API breakage from the experimental 0.1.x series is
likely in various edge-cases.

Highlights include:

- Enriched formula parser to support quoting, and evaluation of formulas involving fields with invalid Python names.
- Added commonly used stateful transformations (identity, center, scale, bs)
- Improved the helpfulness of error messages reported by the formula parser.
- Added support for basic calculus on formulas (useful when taking the gradient of linear models).
- Made it easier to extend Formulaic with additional materializers.
- Many internal improvements to code quality and reliability, including 100% test coverage.
- Added benchmarks for Formulaic against R and patsy.
- Added documentation.
- Miscellaneous other bugfixes and cleanups.

---

## 0.1.2 (6 November 2019)

Performance improvements around the encoding of categorical features.

    Matthew Wardrop (1):
        Improve the performance of encoding operations.


## 0.1.1 (31 October 2019)

No code changes here, just a verification that GitHub CI integration was working.

    Matthew Wardrop (1):
        Update Github workflow triggers.


## 0.1.0 (31 October 2019)

This release added support for keeping track of encoding choices during
model matrix generation, so that they can be reused on similarly structured
data. It also added comprehensive unit testing and CI integration using
GitHub actions.

    Matthew Wardrop (5):
        Add support for stateful transforms (including encoding).
        Fix tokenizing of nested Python function calls.
        Add support for nested transforms that return multiple columns, as well as passing through of materializer config through to transforms.
        Add comprehensive unit testing along with several small miscellaneous bug fixes and improvements.
        Add GitHub actions configuration.

---

## 0.0.1 (1 September 2019)

Initial open sourcing of `formulaic`.

    Matthew Wardrop (1):
        Initial (mostly) working implementation of Wilkinson formulas.
