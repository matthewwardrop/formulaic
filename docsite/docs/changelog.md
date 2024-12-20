For changes since the latest tagged release, please refer to the
[git commit log](https://github.com/matthewwardrop/formulaic/commits/main).

---

## 1.1.1 (20 December 2024)

**New features and enhancements:**

* `Formula.differentiate()` is now considered stable, with
  `ModelMatrix.differentiate()` to follow in a future release.

**Bugfixes and cleanups:**

* Fixed a regression introduced in v1.1.0 regarding ordering of terms in a 
  differentiated formula.

## 1.1.0 (15 December 2024)

**Breaking changes:**

- `Formula` is no longer always "structured" with special cases to handle the
  case where it has no structure. Legacy shims have been added to support old
  patterns, with `DeprecationWarning`s raised when they are used. It is not
  expected to break anyone not explicitly checking whether the `Formula.root` is
  a list instance (which formerly should have been simply assumed) [it is a now
  `SimpleFormula` instance that acts like an ordered sequence of `Term`
  instances].
- The column names associated with categorical factors has changed. Previously,
  a prefix was unconditionally added to the level in the column name like
  `feature[T.A]`, whether nor not the encoding will result in that term acting
  as a contrast. Now, in keeping with `patsy`, we only add the prefix if the
  categorical factor is encoded with reduced rank. Otherwise, `feature[A]` will
  be used instead.
- `formulaic.parsers.types.structured` has been promoted to
  `formulaic.utils.structured`.

**New features and enhancements:**

- `Formula` now instantiates to `SimpleFormula` or `StructuredFormula`, the
  latter being a tree-structure of `SimpleFormula` instances (as compared to
  `List[Term]`) previously. This simplifies various internal logic and makes the
  propagation of formula metadata more explicit.
- Added support for restricting the set of features used by the default formula
  parser so that libraries can more easily restrict the structure of output
  formulae.
- `dict` and `recarray` types are no associated with the `pandas` materializer
  by default (rather than raising), simplifying some user workflows.
- Added support for the `.` operator (which is replaced with all variables not
  used on the left-hand-side of formulae).
- Added **experimental** support for nested formulae of form `[ ... ~ ... ]`.
  This is useful for (e.g.) generating formulae for IV 2SLS.
- Add support for subsettings `ModelSpec[s]` based on an arbitrary
  strictly reduced `FormulaSpec`.
- Added `Formula.required_variables` to more easily surface the expected data
  requirements of the formula.
- Added support for extracting rows dropped during materialization.
- Added cubic spline support for cyclic (`cc`) and natural (`cr`). See
  `formulaic.materializers.transforms.cubic_spline.cubic_spline` for
  more details.
- Added a `lag()` transform.
- Constructing `LinearConstraints` can now be done from a list of strings (for
  increased parity with `patsy`).
- Categorical factors are now preceded with (e.g.) `T.` when they actully
  describe contrasts (i.e. when they are encoded with reduced rank).
- Contrasts metadata is now added to the encoder state via `encode_categorical`;
  which is surfaced via `ModelSpec.factor_contrasts`.
- `Operator` instances now received `context` which is optionally specified by
  the user during formula parsing, and updated by the parser. This is what makes
  the `.` implementation possible.
- Given the generic usefulness of `Structured`, it has been promoted to
  `formulaic.utils`.
- Added explicit support and testing for Python 3.13.

**Bugfixes and cleanups:**

- Fixed nested ordering of `Formula` instance.
- Allow Python tokens to multiple chained parentheses and brackets without using
  quotes as long as the parentheses are balanced.
- Reduced the number of redundant initialisation operations in `Structured`
  instances.
- Fixed pickling `ModelMatrix` and `FactorValues` instances (whenever wrapped
  objects are picklable).
- `basis_spline`: Fixed evaluation involving datasets with null values, and
  disallow out-of-bounds knots.
- Improved robustness of data contexts involving PyArrow datasets.
- We now use the same sentiles throughout the code-base, rather than having
  module specific sentinels in some places.
- Migrated to `ruff` for linting, and updated `mypy` and `pre-commit` tooling.
- Automatic fixes from `ruff` are automatically applied when using
  `hatch run lint:format`.

**Documentation:**

- Fixed and updated docsite build, as well as other minor tweaks.

---

## 1.0.2 (12 July 2024)

**Bugfixes and cleanups:**

- Fix compatibility with `pandas` >=3.
- Fix `mypy` type inference in materializer subclasses.

**Documentation:**

- Add column name extraction to `sklearn` integration example.
- Add section to allow users to indicate their usage of formulaic.

## 1.0.1 (24 December 2023)

**Bugfixes and cleanups:**

- Update package status from "beta" to "production/stable".

## 1.0.0 (24 December 2023)

**Breaking changes:**

- Python tokens are now canonically formatted (see below).
- Methods deprecated during the 0.x series have been removed: `Formula.terms`,
  `ModelSpec.feature_names`, and `ModelSpec.feature_indices`.

**New features and enhancements:**

- Python tokens are now sanitized and canonically formatted to prevent
  ambiguities and better align with `patsy`.
- Added official support for Python 3.12 (no code changes were necessary).
- Added the `hashed` transform for categorically encoding deterministically
  hashed representations of a dataset.

**Bugfixes and cleanups:**

- Fixed transform state not propagating correctly when Python code tokens were
  not canonically formatted.
- Literals in formulae will no longer be silently ignored, and feature scaling
  is now fully supported.
- Improved code parsing and formatting utilities and dropped the requirement for
  `astor` for Python 3.9 and newer.
- Fixed all warnings emitted during unit tests.

**Documentation:**

- Removed incompleteness warnings.
- Added some lightweight developer documents.
- Fixed some broken links.

---

## 0.6.6 (4 October 2023)

This is minor release with one important bugfix.

**Bugfixes and cleanups:**

- Fixes a regression introduced by 0.6.4 whereby missing variables will be
  silently dropped from the formula., rather than raising an exception.

## 0.6.5 (25 September 2023)

This is a minor release with several important bugfixes.

**Bugfixes and cleanups:**

- Fixed intercept terms sorting after other features (by not counting literal
  factors toward the degree of a term).
- Fixed a regression in 0.6.4 around quoted field names in Python evaluations.
- Fixed detection and dropping of null rows in sparse datasets.
- Fixed `poly()` transforms operating on datasets that include null values.
- Arguments can now be passed when running the unit tests using `hatch run tests`.

## 0.6.4 (10 July 2023)

This is a minor release with several new features and cleanups.

**New features and enhancements:**

- Added support for keeping track of the source of variables being used to
  evaluate a formula. Refer to the `ModelSpec` documentation for more details.

**Bugfixes and cleanups:**

- All functions and methods now have type signatures that are statically checked
  during unit testing.
- Removed `OrderedDict` usage, since Python guarantees the orderedness of
  dictionaries in Python 3.7+.
- Suppress terms/factors in model matrices for which the factors evaluate to
  `None`.

## 0.6.3 (26 June 2023)

This is a minor release with a bugfix.

**Bugfixes and cleanups:**

- Fixed a regression introduced in the previous release when materializing
  categorical encodings of variables with no levels.

## 0.6.2 (22 June 2023)

This is a minor release with several bugfixes.

**Bugfixes and cleanups:**

- Fixed issues handling empty data sets in formulae that used categorical
  encoding.
- Added the MIT license to distribution classifiers.

## 0.6.1 (2 May 2023)

This is a minor release with one new feature.

**New features and enhancements:**

- Added support for treating individual categorical features as though they do not span the intercept (useful for intentionally generating over-specified model matrices in e.g. regularized models).

## 0.6.0 (26 Apr 2023)

This is a major release with some important consistency and completeness
improvements. It should be treated as _almost_ being the first release candidate
of 1.0.0, which will land after some small amount of further feature extensions
and documentation improvements. All users are recommended to upgrade.

**Breaking changes:**

Although there are some internal changes to API, as documented below, there are
no breaking changes to user-facing APIs.

**New features and enhancements:**

- Formula terms are now consistently ordered regardless of providence (formulae or
  manual term specification), and sorted according to R conventions by default
  rather than lexically. This can be changed using the `_ordering` keyword to
  the `Formula` constructor.
- Greater compatibility with R and patsy formulae:
  - for patsy: added `standardize`, `Q` and treatment contrasts shims.
  - for patsy: added `cluster_by='numerical_factors` option to `ModelSpec` to enable
    patsy style clustering of output columns by involved numerical factors.
  - for R: added support for exponentiation with `^` and `%in%`.
- Diff and Helmert contrast codings gained support for additional variants.
- Greatly improved the performance of generating sparse dummy encodings when
  there are many categories.
- Context scoping operators (like paretheses) are now tokenized as their own special
  type.
- Add support for merging `Structured` instances, and use this functionality during
  AST evaluation where relevant.
- `ModelSpec.term_indices` is now a list rather than a tuple, to allow direct use when
  indexing pandas and numpy model matrices.
- Add official support for Python 3.11.

**Bugfixes and cleanups:**

- Fix parsing formulae starting with a parenthesis.
- Fix iteration over root nodes of `Structured` instances for non-sequential iterable values.
- Bump testing versions and fix `poly` unit tests.
- Fix use of deprecated automatic casting of factors to numpy arrays during dense
  column evaluation in `PandasMaterializer`.
- `Factor.EvalMethod.UNKNOWN` was removed, defaulting instead to `LOOKUP`.
- Remove `sympy` version constraint now that a bug has been fixed upstream.

**Documentation:**

- Substantial updates to documentation, which is now mostly complete for end-user
  use-cases. Developer and API docs are still pending.

---

## 0.5.2 (17 Sep 2022)

This is a minor patch releases that fixes one bug.

**Bugfixes and cleanups:**

- Fixed alignment between the length of a `Structured` instance and iteration
  over this instance (including `Formula` instances). Formerly the length would
  only count the number of keys in its structure, rather than the number of
  objects that would be yielded during iteration.

## 0.5.1 (9 Sep 2022)

This is a minor patch release that fixes two bugs.

**Bugfixes and cleanups:**

- Fixed generation of string representation of `Formula` objects.
- Fixed generation of `formulaic.__version__` during package build.

## 0.5.0 (28 Aug 2022)

This is a major new release with some minor API changes, some ergonomic
improvements, and a few bug fixes.

**Breaking changes:**

- Accessing named substructures of `Formula` objects (e.g. `formula.lhs`) no
  longer returns a list of terms; but rather a `Formula` object, so that the
  helper methods can remain accessible. You can access the raw terms by
  iterating over the formula (`list(formula)`) or looking up the root node
  (`formula.root`).

**New features and improvements:**

- The `ModelSpec` object is now the source of truth in all `ModelMatrix`
  generations, and can be constructed directly from any supported specification
  using `ModelSpec.from_spec(...)`. Supported specifications include formula
  strings, parsed formulae, model matrices and prior model specs.
- The `.get_model_matrix()` helper methods across `Formula`,
  `FormulaMaterializer`, `ModelSpec` and `model_matrix` objects/helpers
  functions are now consistent, and all use `ModelSpec` directly under the hood.
- When accessing substructures of `Formula` objects (e.g. `formula.lhs`), the
  term lists will be wrapped as trivial `Formula` instances rather than returned
  as raw lists (so that the helper methods like `.get_model_matrix()` can still
  be used).
- `FormulaSpec` is now exported from the top-level module.

**Bugfixes and cleanups:**

- Fixed `ModelSpec` specifications being overriden by default arguments to
  `FormulaMaterializer.get_model_matrix`.
- `Structured._flatten()` now correctly flattens unnamed substructures.

---

## 0.4.0 (10 Aug 2022)

This is a major new release with some new features, greatly improved ergonomics
for structured formulae, matrices and specs, and a few small breaking changes
(most with backward compatibility shims). All users are encouraged to upgrade.

**Breaking changes:**

- `include_intercept` is no longer an argument to `FormulaParser.get_terms`;
  and is instead an argument of the `DefaultFormulaParser` constructor. If you
  want to modify the `include_intercept` behaviour, please use:
  ```python
  Formula("y ~ x", _parser=DefaultFormulaParser(include_intercept=False))
  ```
- Accessing terms via `Formula.terms` is deprecated since `Formula` became a
  subclass of `Structured[List[Terms]]`. You can directly iterate over, and/or
  access nested structure on the `Formula` instance itself. `Formula.terms`
  has a deprecated property which will return a reference to itself in order to
  support legacy use-cases. This will be removed in 1.0.0.
- `ModelSpec.feature_names` and `ModelSpec.feature_columns` are deprecated in
  favour of `ModelSpec.column_names` and `ModelSpec.column_indices`. Deprecated
  properties remain in-place to support legacy use-cases. These will be removed
  in 1.0.0.

**New features and enhancements:**

- Structured formulae (and their derived matrices and specs) are now mutable.
  Internally `Formula` has been refactored as a subclass of
  `Structured[List[Terms]]`, and can be incrementally built and modified. The
  matrix and spec outputs now have explicit subclasses of `Structured`
  (`ModelMatrices` and `ModelSpecs` respectively) to expose convenience methods
  that allow these objects to be largely used interchangeably with their
  singular counterparts.
- `ModelMatrices` and `ModelSpecs` arenow surfaced as top-level exports of the
  `formulaic` module.
- `Structured` (and its subclasses) gained improved integration of nested tuple
  structure, as well as support for flattened iteration, explicit mapping
  output types, and lots of cleanups.
- `ModelSpec` was made into a dataclass, and gained several new
  properties/methods to support better introspection and mutation of the model
  spec.
- `FormulaParser` was renamed `DefaultFormulaParser`, and made a subclass of the
  new formula parser interface `FormulaParser`. In this process
  `include_intercept` was removed from the API, and made an instance attribute
  of the default parser implementation.

**Bugfixes and cleanups:**

- Fixed AST evaluation for large formulae that caused the evaluation to hit the
  recursion limit.
- Fixed sparse categorical encoding when the dataframe index is not the standard
  range index.
- Fixed a bug in the linear constraints parser when more than two constraints
  were specified in a comma-separated string.
- Avoid implicit changing of the sparsity structure of CSC matrices.
- If manually constructed `ModelSpec`s are provided by the user during
  materialization, they are updated to reflect the output-type chosen by the
  user, as well as whether to ensure full rank/etc.
- Allowed use of older pandas versions. All versions >=1.0.0 are now supported.
- Various linting cleanups as `pylint` was added to the CI testing.

**Documentation:**

- Apart from the `.materializer` submodule, most code now has inline
  documentation and annotations.

---

## 0.3.4 (1 May 2022)

This is a backward compatible major release that adds several new features.

**New features and enhancements:**

- Added support for customizing the contrasts generated for categorical
  features, including treatment, sum, deviation, helmert and custom contrasts.
- Added support for the generation of linear constraints for `ModelMatrix`
  instances (see `ModelMatrix.model_spec.get_linear_constraints`).
- Added support for passing `ModelMatrix`, `ModelSpec` and other formula-like
  objects to the `model_matrix` sugar method so that pre-processed formulae can
  be used.
- Improved the way tokens are manipulated for the right-hand-side intercept and
  substitutions of `0` with `-1` to avoid substitutions in quoted contexts.

**Bugfixes and cleanups:**

- Fixed variable sanitization during evaluation, allowing variables with
  special characters to be used in Python transforms; for example:
  `` bs(`my|feature%is^cool`) ``.
- Fixed the parsing of dictionaries and sets within python expressions in the
  formula; for example: `C(x, {"a": [1,2,3]})`.
- Bumped requirement on `astor` to >=0.8 to fix issues with ast-generation in
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

- The minimum supported version of Python is now 3.7 (up from 3.6).
- Moved transform implementations from `formulaic.materializers.transforms` to
  the top-level `formulaic.transforms` module, and ported all existing
  transforms to output `FactorValues` types rather than dictionaries.
  `FactorValues` is an object proxy that allows output types like
  `pandas.DataFrame`s to be used as they normally would, with some additional
  metadata for formulaic accessible via the `__formulaic_metadata__`
  attribute. This makes non-formula direct usage of these transforms much more
  pleasant.
- `~` is no longer a generic formula separator, and can only be used once in a
  formula. Please use the newly added `|` operator to separate a formula into
  multiple parts.

**New features and enhancements:**

- Added support for "structured" formulas, and updated the `~` operator to use
  them. Structured formulas can have named substructures, for example: `lhs`
  and `rhs` for the `~` operator. The representation of formulas has been
  updated to show this structure.
- Added support for context-sensitivity during the resolution of operators,
  allowing more flexible operators to be implemented (this is exploited by the
  `|` operator which splits formulas into multiple parts).
- The `formulaic.model_matrix` syntactic sugar function now accepts `ModelSpec`
  and `ModelMatrix` instances as the "formula" spec, making generation of
  matrices with the same form as previously generated matrices more
  convenient.
- Added the `poly` transform (compatible with R and patsy).
- `numpy` is now always available in formulas via `np`, allowing formulas like
  `np.sum(x)`. For convenience, `log`, `log10`, `log2`, `exp`, `exp10` and
  `exp2` are now exposed as transforms independent of user context.
- Pickleability is now guaranteed and tested via unit tests. Failure to pickle
  any formulaic metadata object (such as formulas, model specs, etc) is
  considered a bug.
- The capturing of user context for use in formula materialization has been
  split out into a utility method `formulaic.utils.context.capture_context()`.
  This can be used by libraries that wrap Formulaic to capture the variables
  and/or transforms available in a users' environment where appropriate.

**Bugfixes and cleanups:**

- Migrated all code to use the Black style.
- Increased unit testing coverage to 100%.
- Fixed mis-alignment in the right- and left-hand sides of formulas if there
  were nulls at different indices.
- Fixed basis spline transforms ignoring state, fixed generated splines for
  large numbers of knots, and fixed specification of knots via non-list
  datatypes.
- Fixed category order being inconsistent if categories are explicitly ordered
  differently in the underlying data.
- Lots of other minor nits and cleanups.

**Documentation:**

- The structure of the docsite has been improved (but is still incomplete).
- The `.parser` and `.utils` modules of Formulaic are now inline documented
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
