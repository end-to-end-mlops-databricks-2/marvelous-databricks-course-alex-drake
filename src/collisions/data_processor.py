import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, input_df: pd.DataFrame):
        pass