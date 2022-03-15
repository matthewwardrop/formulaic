import functools
import os
import sys
import time
from collections import namedtuple

import formulaic
import numpy
import pandas
import patsy
from formulaic import Formula
from uncertainties import ufloat

ALL_TOOLINGS = ["patsy", "formulaic", "formulaic_sparse", "R", "R_sparse"]

formulas = {
    "a": ALL_TOOLINGS,
    "A": ALL_TOOLINGS,
    "a+A": ALL_TOOLINGS,
    "a:A": ALL_TOOLINGS,
    "A+B": ALL_TOOLINGS,
    "a:A:B": ALL_TOOLINGS,
    "A:B:C:D": ALL_TOOLINGS,
    "a*b*A*B": ALL_TOOLINGS,
    "a*b*c*A*B*C": ALL_TOOLINGS,
}


# Utility libraries
TimedResult = namedtuple("TimedResult", ["times", "mean", "stderr"])


def timed_func(func, min_repetitions=7, max_time=20, get_time=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        times = []
        start = time.time()
        while len(times) < min_repetitions and time.time() - start < max_time:
            f_start = time.time()
            result = func(*args, **kwargs)
            f_time = time.time() - f_start

            if get_time:
                f_time = get_time(result, f_time)

            del result

            times.append(f_time)
        return TimedResult(times, numpy.mean(times), numpy.std(times))

    return wrapper


# Generate data for benchmarks
s = 1000000
df = pandas.DataFrame(
    {
        "A": ["a", "b", "c"] * s,
        "B": ["d", "e", "f"] * s,
        "C": ["g", "h", "i"] * s,
        "D": ["j", "k", "l"] * s,
        "a": numpy.random.randn(3 * s),
        "b": numpy.random.randn(3 * s),
        "c": numpy.random.randn(3 * s),
        "d": numpy.random.randn(3 * s),
    }
)
df.head()


@timed_func
def time_patsy(formula):
    return patsy.dmatrix(formula, df)


@timed_func
def time_formulaic(formula):
    return Formula(formula).get_model_matrix(df)


@timed_func
def time_formulaic_sparse(formula):
    return Formula(formula).get_model_matrix(df, output="sparse")


toolings = {
    "patsy": time_patsy,
    "formulaic": time_formulaic,
    "formulaic_sparse": time_formulaic_sparse,
}


try:
    import rpy2
    import rpy2.robjects as robjs

    R_VERSION = rpy2.situation.r_version_from_subprocess()
    R_MATRIX_VERSION = ".".join(str(i) for i in robjs.r("packageVersion('Matrix')")[0])

    robjs.r(
        """
        library(Matrix)
        library(glue)

        s <- 1000000
        df <- data.frame(
            "A"=rep(c('a', 'b', 'c'), s),
            "B"=rep(c('d', 'e', 'f'), s),
            "C"=rep(c('g', 'h', 'i'), s),
            "D"=rep(c('j', 'k', 'l'), s),
            "a"=rnorm(3*s),
            "b"=rnorm(3*s),
            "c"=rnorm(3*s),
            "d"=rnorm(3*s)
        )
    """
    )

    time_R = timed_func(
        robjs.r(
            """
        function (formula) {
            start_time <- Sys.time()
            model.matrix(as.formula(glue("~ ", formula)), df)
            end_time <- Sys.time()
            difftime(end_time, start_time, units="secs")
        }
    """
        ),
        get_time=lambda result, time: result[0],
    )

    time_R_sparse = timed_func(
        robjs.r(
            """
        function (formula) {
            start_time <- Sys.time()
            sparse.model.matrix(as.formula(glue("~ ", formula)), df)
            end_time <- Sys.time()
            difftime(end_time, start_time, units="secs")
        }
    """
        ),
        get_time=lambda result, time: result[0],
    )

    toolings.update(
        {
            "R": time_R,
            "R_sparse": time_R_sparse,
        }
    )

except Exception as e:
    R_VERSION = None
    print(f"Could not set up R benchmarking functions. Error was: {repr(e)}.")


if __name__ == "__main__":
    # Print package versions
    PYTHON_VERSION = sys.version.split("\n")[0].strip()
    print(
        "version information\n"
        f"    python: {PYTHON_VERSION}\n"
        f"        formulaic: {formulaic.__version__}\n"
        f"        patsy: {patsy.__version__}\n"
        f"        pandas: {pandas.__version__}"
    )
    if R_VERSION:
        print(
            f"    R: {R_VERSION}\n"
            f"        model.matrix: (inbuilt into R)\n"
            f"        Matrix (sparse.model.matrix): {R_MATRIX_VERSION}\n"
        )

    # Perform benchmarks
    results = {}
    for formula, config in formulas.items():
        print(formula)
        results[formula] = {}
        for tooling, time_func in toolings.items():
            result = (
                time_func(formula)
                if tooling in config
                else TimedResult(None, numpy.nan, numpy.nan)
            )
            results[formula][tooling] = result
            if not numpy.isnan(result.mean):
                print(
                    f"    {tooling}: {ufloat(result.mean, result.stderr):.2uP} (mean of {len(result.times)})"
                )

    # Dump results into a csv file
    rows = []
    for formula, tooling_results in results.items():
        for tooling, times in tooling_results.items():
            rows.append(
                {
                    "formula": formula,
                    "tooling": tooling,
                    "mean": times.mean,
                    "stderr": times.stderr,
                }
            )
    data = pandas.DataFrame(rows)
    data.to_csv(os.path.join(os.path.dirname(__file__), "benchmarks.csv"))
