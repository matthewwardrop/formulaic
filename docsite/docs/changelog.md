For changes since the latest tagged release, please refer to the
[git commit log](https://github.com/matthewwardrop/formulaic/commits/main).

---

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
