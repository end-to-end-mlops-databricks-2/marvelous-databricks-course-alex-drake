# Databricks notebook source
import mlflow

from mlflow.models import infer_signature
from sklearn.metrics import log_loss, roc_auc_score

from reservations.config import Config
from reservations.data_loader import DataLoaderUC
from reservations.model_lightgbm import CustomLGBModel, CustomWrapper

# load configuration
config = Config.from_yaml(config_path="project_config.yml")

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
if mlflow.get_experiment(config.experiment_name) is None:
    mlflow.create_experiment(
        name=config.experiment_name,
        artifact_location=f"{config.schema_name}/models"
    )
experiment_id = mlflow.set_experiment(
    config.experiment_name
    ).experiment_id

# COMMAND ----------

# could use this later for e.g. hyperopt
params = None

# run experiment

with mlflow.start_run(
    experiment_id=experiment_id,
    tags={"tag_key":"tag_value"}
    ) as run:

    # may come back to later for nested runs / getting best run
    run_id = run.info.run_id

    if params is None:
        parameters = config.parameters
    else:
        parameters = params

    classifier = CustomLGBModel(**parameters)
    classifier.train(X_train, y_train)

    y_pred = classifier.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred)

    metrics = {
        "val_roc_auc": roc_auc,
        "val_log_loss": ll
    }

    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)

    signature = infer_signature(
        model_input=X_test,
        model_output=y_pred
    )

    WrappedModel = CustomWrapper(classifier)

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=WrappedModel,
        signature=signature
    )
