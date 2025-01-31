import unittest

from unittest.mock import MagicMock
from pyspark.sql import SparkSession

from pandas import CategoricalDtype
from pandas.api.types import is_numeric_dtype
from tests.base_test import BaseTest

from src.reservations.data_processor import DataProcessor


class TestDataProcessor(BaseTest):
    """
    Test DataProcessor functionality
    """

    def test_data_processor_loads_df(self):
        """
        Test that the data processor loads a dataframe from the 
        config file when input dataframe is set to None
        """
        data_processor = DataProcessor(None, config=self.config)
        self.assertIsNotNone(
            data_processor.df,
            "Imported dataframe should not be None"
            )
        
    def test_preprocessor(self):
        """
        Test that the preprocessor handles the data correctly
        """
        data_processor = DataProcessor(None, config=self.config)
        data_processor.preprocess()
        
        a_num_feature = self.config.num_features[0]
        a_cat_feature = self.config.cat_features[0]
        
        self.assertTrue(
            is_numeric_dtype(data_processor.df[a_num_feature]),
            f"Expected numeric dtype for column {a_num_feature}"
        )
        
        self.assertIsInstance(
            data_processor.df[a_cat_feature].dtype,
            CategoricalDtype,
            f"Expected categorical dtype for column {a_cat_feature}"
        )

    def test_data_splitter(self):
        """Test that the dataframe can be split into train and test
        dataframes.
        """
        data_processor = DataProcessor(None, config=self.config)
        data_processor.preprocess()
        data_processor.split_data(test_size=0.3, random_state=10)

        self.assertIsNotNone(
            data_processor.train_df,
            "Training data should not be None. Check the preprocessing steps."
        )

        self.assertIsNotNone(
            data_processor.test_df,
            "Test data should not be None. Check the preprocessing steps."
        )
        
    def test_write_to_databricks(self):
        """
        Placeholder for testing writing to DB
        """
        pass
