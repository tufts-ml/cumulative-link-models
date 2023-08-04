import pytest

from sklearn.utils.estimator_checks import check_estimator

from src.clm_sklearn import OrdinalRegression


@pytest.mark.parametrize(
    "estimator",
    [OrdinalRegression()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
