import pytest

from sklearn.utils.estimator_checks import check_estimator

from autopycoin import TemplateEstimator
from autopycoin import TemplateClassifier
from autopycoin import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
