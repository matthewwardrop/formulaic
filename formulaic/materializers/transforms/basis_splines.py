"""
Modified version of code originally from Zepid, Copyright (c) 2018 Paul Zivich.
https://github.com/pzivich/zEpid/blob/master/LICENSE.txt
"""
import numpy as np
import pandas as pd
from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def basis_splines(data, n_knots=3, knots=None, term=1, restricted=False, state=None):
    """
    Creates spline dummy variables based on either user specified knot locations or automatically
    determines knot locations based on percentiles. Options are available to set the number of knots,
    location of knots (value), term (linear, quadratic, etc.), and restricted/unrestricted.

    Parameters
    --------------
    data:
    n_knots : integer, optional
        Number of knots requested. Options for knots include any positive integer if the location of knots are
        specified. If knot locations are not specified, n_knots must be an integer between 1 to 7. Default is 3 knots.
        Ignored if `knots` is specified.
    knots : list, optional
        Location of specified knots in a list. To specify the location of knots, put desired numbers for knots into a
        list. Be sure that the length of the list is the same as the specified number of knots. Default is None, so
        that the function will automatically determine knot locations without user specification
    term : integer, float, optional
        High order term for the spline terms. To calculate a quadratic spline change to 2, cubic spline
        change to 3, etc. Default is 1, i.e. a linear spline
    restricted : bool, optional
        Whether to return a restricted spline. Note that the restricted spline returns one less column than the number
        of knots. An unrestricted spline returns the same number of columns as the number of knots. Default is False,
        providing an unrestricted spline

    """
    if knots is None:
        if n_knots == 1:
            knots = [50]
        elif n_knots == 2:
            knots = [100 / 3, 200 / 3]
        elif n_knots == 3:
            knots = [5, 50, 95]
        elif n_knots == 4:
            knots = [5, 35, 65, 95]
        elif n_knots == 5:
            knots = [5, 27.5, 50, 72.5, 95]
        elif n_knots == 6:
            knots = [5, 23, 41, 59, 77, 95]
        elif n_knots == 7:
            knots = [2.5, 1100 / 60, 2600 / 75, 50, 7900 / 120, 4900 / 60, 97.5]
        else:
            raise ValueError(
                "When the knot locations are not pre-specified, the number of specified knots must be"
                " an integer between 1 and 7")
        pts = np.percentile(data, q=knots).tolist()
    else:
        pts = sorted(knots)
        n_knots = len(pts)

    def bs_transform(x):
        columns = np.arange(n_knots)
        x = np.asarray(x)
        V = np.empty((x.shape[0], len(pts)))

        for i, pt in enumerate(pts):
            V[:, i] = np.where(x > pt, (x - pt) ** term, 0)
            V[:, i] = np.where(pd.isnull(x), np.nan, V[:, i])
        if restricted is False:
            return pd.DataFrame(V, columns=columns)

        else:
            for i, pt in enumerate(pts):
                V[:, i] = np.where(x > pt, V[:, i] - V[:, -1], 0)
                V[:, i] = np.where(pd.isnull(x), np.nan, V[:, i])
            return pd.DataFrame(V[:, :-1], columns=columns[:-1])

    if 'bs_transform' not in state:
        state['bs_transform'] = bs_transform

    encoded = dict(state['bs_transform'](data))
    encoded.update({
        '__format__': "{name}[{field}]",
        })

    return encoded
