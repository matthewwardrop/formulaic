For the latest changes, please refer to the git log: https://github.com/matthewwardrop/formulaic/commits/master.

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


## 0.0.1 (1 September 2019)

Initial open sourcing of `formulaic`.

    Matthew Wardrop (1):
        Initial (mostly) working implementation of Wilkinson formulas.
