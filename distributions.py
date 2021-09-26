import theano.tensor as tt
import pymc3 as pm

from pymc3.theanof import floatX, intX
from pymc3.util import get_var_name


def betaln(a, b):
    return tt.gammaln(a) + tt.gammaln(b) - tt.gammaln(a + b)


def beta(a, b):
    return tt.gamma(a) * tt.gamma(b) / tt.gamma(a, b)


def log_beta_pdf(x, a, b):
    return (a - 1) * tt.log(x) + (b - 1) * tt.log(1 - x) - betaln(a, b)


def beta_geom_llh(x, a, b):
    return tt.gammaln(a + 1) + tt.gammaln(x + b - 1) - tt.gammaln(a + x + b) - betaln(a, b)


def censored_beta_geom_llh(x, a, b):
    return betaln(a, b + x) - betaln(a, b)


def bg_pdf(x, a, b):
    return beta(a + 1, x + b - 1) / beta(a, b)


class RightCensoredBetaGeometric(pm.Discrete):
    """
    Pymc implementation of:
    Fader, Peter and Hardie, Bruce, How to Project Customer Retention (May 2006).
    Available at SSRN: https://ssrn.com/abstract=801145.
    or http://dx.doi.org/10.2139/ssrn.801145

    Expects data at individual level, value should be an array containing
    for each customer in your dataset the number of renewals. Conversely
    censored should be a 0/1 array of the same size specifying if the user
    full lifetime history had been observed (where 1=not observed, i.e. censored).
    """

    def __init__(self, a, b, censored, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = tt.as_tensor_variable(floatX(a))
        self.b = tt.as_tensor_variable(floatX(b))
        self.censored = tt.as_tensor_variable(intX(censored))
        self.mode = 1

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        a = dist.a
        b = dist.b
        name = r'\text{%s}' % name
        return r'${} \sim \text{{RCBetaGeometric}}(\mathit{{a}}={}, \mathit{{b}}={})$'.format(name,
                                                                                              get_var_name(a),
                                                                                              get_var_name(b))

    def geo_logp(self, value):
        return beta_geom_llh(value, self.a, self.b)

    def logp(self, value):
        censored = self.censored
        geo_llh = self.geo_logp(value)
        censored_geo_llh = censored_beta_geom_llh(value, self.a, self.b)
        return tt.switch(tt.eq(censored, 1), censored_geo_llh, geo_llh)


class WeightedRightCensoredBetaGeometric(RightCensoredBetaGeometric):
    """
    Same as `RightCensoredBetaGeometric` but works over cohorts.
    So your data should be in the shape of:

    uncensored: the number of users by cohort for which we have observed churn
    censored: the number of users by cohort for which we have not observed churn

    Note:
        (1) data could also be not monotonically decreasing (you might have more
        people for which you have observed churn at 3 renewals than at 4).
        (2) if you have data like in BHF original paper, censored will look like an array of
        zeros with a final value that is the one at the end of the table, while uncensored will
        be the rest of the table with a zero instead of the last value. Nevertheless this grouped data
        will give you a worst estimate.
    """

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncensored = tt.as_tensor_variable(intX(n))

    def logp(self, value):
        n = self.uncensored
        censored = self.censored
        geo_llh = self.geo_logp(value)
        censored_geo_llh = censored_beta_geom_llh(value, self.a, self.b)
        return censored * censored_geo_llh + n * geo_llh
