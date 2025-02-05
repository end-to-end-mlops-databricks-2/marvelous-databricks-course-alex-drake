"""
LightGBM Model generation and testing for hotel reservation
cancellation project
"""

import pandas as pd

import mlflow
import mlflow.lightgbm

from mlflow.models import infer_signature
from sklearn.metrics import log_loss, roc_auc_score
from lightgbm import LGBMClassifier

from reservations.config import Config


class CustomLGBModel:
    """
    A custom class to create a LightGBM model

    :param config: Configuration object for data cleaning.
    """
    def __init__(self, config: Config) -> None:
        self.config = config
        self.schema = f"{config.catalog_name}.{config.schema_name}"

        self.params = config.parameters
        self.experiment_id = self._get_experiment_id()
        self.model = LGBMClassifier(**self.params)

    def _get_experiment_id(self) -> str:
        """
        Gets the experiment ID for MLFlow
        """
        if mlflow.get_experiment(self.config.experiment_name) is None:
            mlflow.create_experiment(
                name=self.config.experiment_name,
                artifact_location=f"{self.schema}/models"
            )
        experiment_id = mlflow.set_experiment(
            self.config.experiment_name
            ).experiment_id

        return experiment_id

    def train(self, X, y):
        """
        Train function for LGBMClassifier

        :param X: a pd.DataFrame of training data
        :param y: a pd.Series of target data
        """
        self.model.fit(X, y)
        return self

class CustomerWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom wrapper for trained model

    Can be used to alter prediction method
    i.e. switch to predict_proba if relevant
    """
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        """
        Run predictions on the input data
        """
        return self.model.predict(model_input)
