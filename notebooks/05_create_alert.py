# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the percentage of MAE being higher than 7000

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

sources = w.data_sources.list()

alert_query = """
SELECT
"""