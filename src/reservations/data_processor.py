import datetime

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from reservations.config import Config


class DataProcessor:
    """Data Processor

    Pre-process data for use in an ML project.
    Data can be saved to a designated catalog
    within Databricks
    """
    def __init__(self, input_df: pd.DataFrame, config: Config):
        self.config = config

        if input_df is not None:
            self.df = input_df
        else:
            self.get_data_from_config()

        self.train_df = None
        self.test_df = None

    def get_data_from_config(self):
        """Retrieve input data from the config dataset"""
        file_loc = self.config.data
        self.df = pd.read_csv(file_loc)

    def preprocess(self):
        """Preprocess the input DataFrame"""
        target = self.config.target
        self.df[target] = self.df[target].apply(
            lambda x: 0 if x == 'Not_Canceled' else 1
            )

        # treat the numerical features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # treat the categorical features
        cat_features = self.config.cat_features
        encoded_features = []
        for col in cat_features:
            new_col = col+"_encoded"
            self.df[new_col] = self.df[col].astype("category").cat.codes
            encoded_features.append(new_col)

        # additional date features
        date_features = ['dayofyear', 'arrival_week', 'dayofweek']

        self.df['year'] = self.df['arrival_year']
        self.df['month'] = self.df['arrival_month']
        self.df['day'] = self.df['arrival_date']

        self.df['date'] = pd.to_datetime(
            self.df[['year', 'month', 'day']],
            errors='coerce'
        )

        self.df['dayofyear'] = self.df['date'].dt.dayofyear
        self.df['arrival_week'] = self.df['date'].dt.isocalendar(). \
            week.astype(float)
        self.df['dayofweek'] = self.df['date'].dt.dayofweek

        # retain the desired features
        id_col = self.config.id_column
        self.df[id_col] = self.df[id_col].astype("str")

        req_columns = [id_col] + num_features + cat_features + \
            encoded_features + date_features + [target]

        self.df = self.df[req_columns]

    def split_data(self, test_size=0.3, random_state=42):
        """Split the dataframe into train and test sets"""
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state
            )

    def save_df_to_catalog(self, save_df: pd.DataFrame, table_name: str,
                           spark: SparkSession):
        """Save table to Databricks"""

        catalog_destination = (
            f"{self.config.catalog_name}."
            f"{self.config.schema_name}."
            f"{table_name}"
            )

        spark_df = spark.createDataFrame(save_df).withColumn(
            "update_timestamp_utc",
            to_utc_timestamp(current_timestamp(), "UTC")
        )

        spark_df.write.mode("append").saveAsTable(catalog_destination)

        spark.sql(
            f"""
            ALTER TABLE {catalog_destination}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
            """
        )

    def save_traing_data_to_catalog(self):
        """Save training data to Databricks"""

        if self.train_df is not None:
            self.save_df_to_catalog(
                save_df=self.train_df,
                table_name='train_set',
                spark=SparkSession.builder.getOrCreate()
                )

        if self.test_df is not None:
            self.save_df_to_catalog(
                save_df=self.test_df,
                table_name='test_set',
                spark=SparkSession.builder.getOrCreate()
            )

    def make_synthetic_data(self, drift=False, num_rows=10):
        """
        Generates synthetic data based on the input DataFrame
        """
        synthetic_data = pd.DataFrame()

        for column in self.df.columns:
            if column == self.config.id_column:
                randint = np.random.randint(40000, 99999, num_rows)
                id_col = [f"INN{i}" for i in randint]
                synthetic_data[column] = id_col

            if pd.api.types.is_numeric_dtype(self.df[column]):
                synthetic_data[column] = np.random.randint(
                    self.df[column].min(),
                    self.df[column].max(),
                    num_rows
                )
            elif pd.api.types.is_string_dtype(self.df[column]):
                synthetic_data[column] = np.random.choice(
                    self.df[column].unique(),
                    num_rows,
                    p=self.df[column].value_counts(normalize=True)
                )
            else:
                synthetic_data[column] = np.random.choice(
                    self.df[column],
                    num_rows
                )

        if drift:
            skew_features = [
                "no_of_previous_cancellations",
                "no_of_previous_bookings_not_canceled"
                ]
            for feature in skew_features:
                synthetic_data[feature] = synthetic_data[feature] * 2

            synthetic_data['arrival_year'] = np.random.randint(
                self.df['arrival_year'].max()+1,
                self.df['arrival_year'].max()+3,
                num_rows
            )

        return synthetic_data
