from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, ArrayType
)
import logging
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from databricks.sdk.errors import NotFound


def create_or_refresh_monitoring(config, spark, workspace):
    pass