# Databricks notebook source

from reservations.data_processor import DataProcessor
from reservations.config import Config

# load configuration
config = Config.from_yaml(config_path="project_config.yml")

print(f"Configuration contents: {config}")

# COMMAND ----------
# Initialise the DataProcesser

data_processor = DataProcessor(input_df=None, config=config)

# Process the data
data_processor.preprocess()
data_processor.df.head(5)

# COMMAND ----------
# Split the data
data_processor.split_data(
    test_size=0.3,
    random_state=42
    )

print(f"Training set: {data_processor.train_df.shape}")
print(f"Test set: {data_processor.test_df.shape}")

# COMMAND ----------
# Save training data to the catalog

data_processor.save_traing_data_to_catalog()
