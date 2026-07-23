# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# COMMAND ----------

import gc
import pandas as pd
from pyspark.ml.feature import StringIndexer


def clear_ml_cache(globals_dict, model_vars=None):
    """Release Spark Connect ML model references to stay within the Serverless v5
    session cache limit (1 GB). Pass globals() from the calling notebook.
    Optionally pass a list of variable names to release; defaults to the standard
    pipeline variables used in this course."""
    if model_vars is None:
        model_vars = ["pipeline_model", "loaded_pipeline", "pipeline"]
    released = [v for v in model_vars if v in globals_dict]
    for v in released:
        del globals_dict[v]
    if released:
        gc.collect()
        print(f"ML cache cleanup: released {len(released)} model reference(s): {released}")
    else:
        print("ML cache cleanup: no prior model references found.")


def build_string_indexer_model(spark, categorical_cols, telco_labels):
    """Return a fitted StringIndexerModel built from a fixed label vocabulary.

    Serverless v5 (Spark Connect ML) does not support StringIndexerModel.fromLabels().
    This function fits StringIndexer on a tiny synthetic DataFrame that contains every
    known label value. StringIndexerModel stores only the vocabulary (label → index
    mapping), so a 4-row fit produces an equally small, deterministic model without
    scanning the full training dataset."""
    categorical_cols_indexed = [c + "_index" for c in categorical_cols]
    max_labels = max(len(v) for v in telco_labels.values())
    synthetic_rows = [
        {c: telco_labels[c][i % len(telco_labels[c])] for c in categorical_cols}
        for i in range(max_labels)
    ]
    synthetic_sdf = spark.createDataFrame(
        pd.DataFrame(synthetic_rows)
    ).select(categorical_cols)
    string_indexer = StringIndexer(
        inputCols=categorical_cols,
        outputCols=categorical_cols_indexed,
        stringOrderType="alphabetAsc",
        handleInvalid="keep"
    )
    return string_indexer.fit(synthetic_sdf)


def check_model_size(model, temp_path, size_limit=268_435_456):
    """Verify a fitted model's serialized size is within the Serverless v5 256 MB
    per-model limit. Saves the model to temp_path, recursively sums file sizes, and
    raises AssertionError with a clear message if the limit is exceeded."""
    def _dir_size(path):
        total = 0
        try:
            for item in dbutils.fs.ls(path):
                if item.name.endswith("/"):
                    total += _dir_size(item.path)
                else:
                    total += item.size
        except Exception:
            pass
        return total

    model.write().overwrite().save(temp_path)
    size = _dir_size(temp_path)
    print(f"Model size : {size:>15,} bytes  ({size / 1024**2:.1f} MB)")
    print(f"Limit      : {size_limit:>15,} bytes  ({size_limit // 1024**2} MB)")
    assert size <= size_limit, (
        f"Model ({size / 1024**2:.1f} MB) exceeds the Serverless v5 "
        f"{size_limit // 1024**2} MB per-model limit. Simplify the pipeline to reduce model size."
    )
    print(f"✅ Model size is within the Serverless v5 limit.")
    return size

# COMMAND ----------

@DBAcademyHelper.add_init
def initialize_uc(self):

    table_name = 'telco_table'
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE SCHEMA {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    print(f'Using catalog {DA.catalog_name} and schema {DA.schema_name}.')

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()