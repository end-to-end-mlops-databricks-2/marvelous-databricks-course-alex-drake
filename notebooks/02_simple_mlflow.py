# Databricks notebook source

!pip install /Volumes/mlops_dev/aldrake8/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------
import mlflow

from mlflow.models import infer_signature
from sklearn.metrics import log_loss, roc_auc_score

from reservations.config import Config, Tags
from reservations.data_loader import DataLoaderUC
from reservations.models.model_lightgbm import CustomLGBModel

# load configuration
config = Config.from_yaml(config_path="project_config.yml")
tags = Tags(**{"git_sha": "abcd1234", "branch": "dev"})

print(f"Configuration contents: {config}")

# COMMAND ----------

# load data
data_loader = DataLoaderUC(config=config)
X_train = data_loader.train_set.drop(columns=[config.target])
y_train = data_loader.train_set[config.target]

X_test = data_loader.test_set.drop(columns=[config.target])
y_test = data_loader.test_set[config.target]

print(f"Train shape: {X_train.shape}")

# COMMAND ----------

# set experiment
mlflow.set_tracking_uri('databricks')
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------
# initialise model and train
custom_model = CustomLGBModel(
    tags=tags, config=config
)

custom_model.train(X_train, y_train)

# COMMAND ----------
custom_model.log_model(
    X_test, y_test,
    code_paths=["../src/reservations/"]
)

# COMMAND ----------
# Get model metadata
custom_model.retreive_current_run_metadata()

# COMMAND ----------
# Register the model
custom_model.register_model()

# COMMAND ----------
# Show predictions on test set

predictions = custom_model.load_latest_model_and_predict(X_test)
predictions