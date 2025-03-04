import unittest
import pandas as pd

from tests.base_test import BaseTest

from reservations.models.model_lightgbm import CustomLGBModel


class TestCustomLGBModel(BaseTest):
    """
    Test CustomLGBModel functionality
    """
    def test_model_init(self):
        """
        Test that the class initiatlises with
        a model object
        """
        classifier = CustomLGBModel(tags=self.tags, config=self.config)
        self.assertIsNotNone(
            classifier.model,
            "Model cannot be none"
        )
        self.assertEqual(
            classifier.model.learning_rate,
            self.config.parameters["learning_rate"],
            "Expected learning parameters to match"
        )

    def test_model_training(self):
        """
        Test that the model training function
        does indeed train a model
        """
        data = {
            'feat1': [10,20,30,40],
            'feat2': [5,10,15,20],
            'Booking_ID': ['a','b','c','d'],
            'target': [0,1,0,1]
        }
        df = pd.DataFrame(data)
        X = df.drop('target', axis=1)
        y = df['target']

        classifier = CustomLGBModel(tags=self.tags, config=self.config)
        classifier.train(X, y)

        self.assertIsNotNone(
            classifier.model.best_score_,
            "Model not trained"
        )