For the latest changes, please refer to the git log: https://github.com/matthewwardrop/formulaic/commits/master.

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
