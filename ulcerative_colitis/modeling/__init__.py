from ulcerative_colitis.modeling.classification_head import ClassificationHead
from ulcerative_colitis.modeling.ordinal_regression_head import LogisticCumulativeLink
from ulcerative_colitis.modeling.regression_head import RegressionHead


__all__ = ["ClassificationHead", "RegressionHead", "LogisticCumulativeLink"]
