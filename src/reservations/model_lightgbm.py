"""
LightGBM Model generation and testing for hotel reservation
cancellation project
"""

import mlflow
import mlflow.lightgbm

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
        self.model = LGBMClassifier(**self.params)

    def train(self, X, y):
        """
        Train function for LGBMClassifier

        :param X: a pd.DataFrame of training data
        :param y: a pd.Series of target data
        """
        self.model.fit(X, y)
        return self

    def predict(self, model_input):
        """
        Standard predict
        """
        return self.model.predict(model_input)

    def predict_proba(self, model_input):
        """
        Predict probabilities
        """
        return self.model.predict_proba(model_input)


class CustomWrapper(mlflow.pyfunc.PythonModel):
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
