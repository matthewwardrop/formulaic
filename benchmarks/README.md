# Formulaic Benchmarks

These benchmarks compare the performance of `formulaic` against the existing
formula parsers for Python ([`patsy`](https://patsy.readthedocs.io/)) and R
([`model.matrix`](https://www.rdocumentation.org/packages/stats/versions/3.6.1/topics/model.matrix)
/
[`sparse.model.matrix`](https://www.rdocumentation.org/packages/Matrix/versions/1.2-17/topics/sparse.model.matrix)) when interpreting Wilkinson formulas and
generating the appropriate model matrices. These benchmarks are somewhat
synthetic, and target large data sizes where performance is more critical. As
such, all of the formula-to-model-matrix transforms are tested on a data frame
with three million rows represented as a Pandas or R dataframe. For the time
being, only CPU performance (as compared to memory utilization) is considered.

To run these benchmarks, install `formulaic` and the benchmarking dependencies using `pip install formulaic[benchmarks]` and then run in a checked out copy of this repository:
```
python <formulaic_repo>/benchmarks/benchmark.py
```

**Note:** This will not install R or the required R dependency `Matrix`. This
benchmark will gracefully skip R benchmarks if these are not found.

You can run the standard visualization code using:
```
python <formulaic_repo>/benchmarks/plot.py
```

## Results

On a ThinkPad T480s with a Intel® Core™ i7-8650U CPU @ 1.90GHz and 24 GB of DDR4
RAM, this benchmark yields the following results:

![Benchmarking Results](benchmarks.png)

```
version information
    python: 3.7.5 (default, Oct 25 2019, 15:51:11)
        formulaic: 0.1.2
        patsy: 0.5.1
        pandas: 0.25.2
    R: R version 3.6.1 (2019-07-05) -- "Action of the Toes"
        model.matrix: (inbuilt into R)
        Matrix (sparse.model.matrix): 1.2.17

a
    patsy: 0.0452±0.0045 (mean of 7)
    formulaic: 0.0184±0.0011 (mean of 7)
    formulaic_sparse: 0.322±0.027 (mean of 7)
    R: 0.219±0.039 (mean of 7)
    R_sparse: 0.377±0.055 (mean of 7)
A
    patsy: 6.941±0.028 (mean of 3)
    formulaic: 0.1153±0.0066 (mean of 7)
    formulaic_sparse: 0.445±0.013 (mean of 7)
    R: 0.218±0.034 (mean of 7)
    R_sparse: 0.629±0.046 (mean of 7)
a+A
    patsy: 6.943±0.023 (mean of 3)
    formulaic: 0.1102±0.0029 (mean of 7)
    formulaic_sparse: 0.487±0.012 (mean of 7)
    R: 0.291±0.055 (mean of 7)
    R_sparse: 0.833±0.098 (mean of 7)
a:A
    patsy: 7.156±0.035 (mean of 3)
    formulaic: 0.138±0.010 (mean of 7)
    formulaic_sparse: 0.5165±0.0081 (mean of 7)
    R: 0.267±0.058 (mean of 7)
    R_sparse: 0.611±0.062 (mean of 7)
A+B
    patsy: 14.36±0.13 (mean of 2)
    formulaic: 0.2013±0.0034 (mean of 7)
    formulaic_sparse: 0.592±0.019 (mean of 7)
    R: 0.244±0.044 (mean of 7)
    R_sparse: 0.986±0.035 (mean of 7)
a:A:B
    patsy: 15.091±0.016 (mean of 2)
    formulaic: 0.3369±0.0095 (mean of 7)
    formulaic_sparse: 0.837±0.030 (mean of 7)
    R: 0.424±0.033 (mean of 7)
    R_sparse: 2.48±0.21 (mean of 7)
A:B:C:D
    patsy: 40.41527056694031±0 (mean of 1)
    formulaic: 0.794±0.016 (mean of 7)
    formulaic_sparse: 1.592±0.018 (mean of 7)
    R: 1.361±0.045 (mean of 7)
    R_sparse: 12.58±0.27 (mean of 2)
a*b*A*B
    patsy: 18.71±0.16 (mean of 2)
    formulaic: 0.589±0.015 (mean of 7)
    formulaic_sparse: 1.125±0.025 (mean of 7)
    R: 0.649±0.013 (mean of 7)
    R_sparse: 7.50±0.18 (mean of 3)
a*b*c*A*B*C
    patsy: 55.63201975822449±0 (mean of 1)
    formulaic: 3.249±0.016 (mean of 7)
    formulaic_sparse: 3.791±0.018 (mean of 6)
    R: 3.143±0.042 (mean of 7)
    R_sparse: (1.0259734272956849±0)×10² (mean of 1)
```
