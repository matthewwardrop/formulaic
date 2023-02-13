THanks, Matthew!

I was actually able to figure it out myself yesterday from a previous issue about this topic. So I guess it kind of boils down to the documentation :/.

If I may recommend and show how I want to use this is to have a sklean component:

```python
from sklearn.base import BaseEstimator, TransformerMixin
from formulaic import Formula, model_matrix

class FormulaicTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, formula):
        self.formula = formula

    def fit(self, X, y = None):
        """Fits the estimator"""
        self._trans = model_matrix(self.formula, X).model_spec.rhs
        return self

    def transform(self, X, y= None):
        """Fits the estimator"""
        X_ = self._trans.get_model_matrix(X)
        return X_


pipe = Pipeline([
    ("formula", FormulaicTransformer("(bs(yday, df=12) + wday + num_date")),
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])
```

As this persists the design info and can be pickled. It may be used as a proper sklearn component!
This is a badass feature!