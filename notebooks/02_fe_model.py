# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession

from reservations.config import Config, Tags
from reservations.models.feature_model import FeatureLookUpModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = Config.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd1234", "branch": "dev"}
tags = Tags(**tags_dict)


# COMMAND ----------
# Init model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
# Create feature table and define feature function
fe_model.create_feature_table()
fe_model.define_feature_function()

# COMMAND ----------
# load data and engineer features
fe_model.load_data()
fe_model.feature_engineering()

# COMMAND ----------
# train the model
fe_model.train()

# COMMAND ----------
# register the model
fe_model.register_model()

# COMMAND ----------
# run predictions on the latest prod model
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set"
).limit(10)

# drop the feature lookup columns and target
X_test = test_set.drop(
    "no_of_previous_cancellations","no_of_previous_bookings_not_canceled", config.target
)

# COMMAND ----------
predictions = fe_model.load_latest_model_and_predict(X_test)
print(f"Predictions: {predictions}")