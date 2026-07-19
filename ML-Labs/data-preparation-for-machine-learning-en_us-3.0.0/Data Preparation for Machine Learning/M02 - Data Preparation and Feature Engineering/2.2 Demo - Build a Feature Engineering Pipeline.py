# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Build a Feature Engineering Pipeline
# MAGIC
# MAGIC In this demo, we will build a feature engineering pipeline that performs data loading, imputation, transformation, and encoding of categorical features. The pipeline will be applied to training and testing datasets, ensuring consistency in data preprocessing. Finally, we will save the pipeline for future reuse, allowing efficient and reproducible data preparation for machine learning.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Build a structured feature engineering pipeline that includes multiple preprocessing steps.
# MAGIC * Encode categorical features using **StringIndexer** and **OneHotEncoder**.
# MAGIC * Impute missing numerical values using `Imputer` and scale features with `StandardScaler`.
# MAGIC * Assemble transformed numerical and encoded categorical features into a single feature vector.
# MAGIC * Prepare the target column for machine learning by converting it to a numeric label.
# MAGIC * Apply the feature engineering pipeline to both training and test datasets.
# MAGIC * Save a data preparation and feature engineering pipeline to Unity Catalog for potential future use.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC
# MAGIC <div style="border-left: 4px solid #F44336; background: #FFEBEE; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #C62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Serverless Compute, Version 5</strong> - <a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976D2; text-decoration: underline;">How to select an environment version</a></li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V5</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ⚠️ SERVERLESS: RESTART YOUR SESSION IF NEEDED
# MAGIC
# MAGIC <div style="border-left: 4px solid #FF9800; background: #FFF3E0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #E65100; font-size: 1.1em;">Restart the Serverless Session</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Because this notebook runs on <strong>Serverless</strong>, you may occasionally need to start a fresh session for state to reset cleanly. To do this, either:</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Restart the session</strong> — open the compute drop-down (top right) and choose <strong>New session</strong> to begin a new session, <em>or</em></li>
# MAGIC <li><strong>Terminate and start</strong> — <strong>terminate</strong> the current Serverless compute, then <strong>start</strong> it again before re-running the cells.</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #F44336; background: #FFEBEE; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #C62828; font-size: 1.1em;">If you hit this error</strong>
# MAGIC <p style="margin: 8px 0 0 0;"><code style="background: #FCE4EC; color: #B71C1C; padding: 2px 6px; border-radius: 3px;">[CONNECT_ML.MODEL_SIZE_EXCEEDED_EXCEPTION] Spark Connect ML error: The fitted or loaded model size is about … bytes. Fit or load a model smaller than 268435456 bytes. SQLSTATE: 54000</code></p>
# MAGIC <p style="margin: 12px 0 0 0; color: #333;"><strong>Fix:</strong> <strong>Restart the session</strong> (compute drop-down → <strong>New session</strong>) <em>or</em> <strong>terminate and start</strong> the Serverless compute, then re-run the notebook cells from the top.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Run the following cell to configure your working environment for this course.
# MAGIC
# MAGIC This setup will:
# MAGIC - Initialize the `DA` object (Databricks Academy helper)
# MAGIC - Configure your **default catalog** and **schema**
# MAGIC - Provision any supporting configuration needed for this demo
# MAGIC
# MAGIC **NOTE:** The `DA` object is only available in Databricks Academy courses.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-2.2u

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains **variables such as your username, catalog name, schema name, working directory, and dataset locations**. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# Release any stale ML model references from previous runs (Serverless v5 — 1 GB session cache limit)
clear_ml_cache(globals())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data Preparation
# MAGIC
# MAGIC Before constructing the feature engineering pipeline, we need to ensure the dataset is consistent and properly formatted. This includes handling data types, addressing missing values, and preparing the dataset for further transformations. The `Telco Customer Churn` dataset will be used for this process.
# MAGIC
# MAGIC **Steps in Data Preparation:**
# MAGIC 1. Load the dataset into a Spark DataFrame.
# MAGIC 1. Split the dataset into training and testing sets.
# MAGIC 1. Convert Integer and Boolean columns to Double to ensure compatibility with Spark ML.
# MAGIC 1. Handle missing values by identifying and imputing them in:
# MAGIC     - Numeric columns
# MAGIC     - String columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading the Dataset
# MAGIC We start by loading the dataset from the specified file path using Spark. Note that we will use `.option("nullValue", " ")` since this particular CSV has empty string values that need to be read as null. This will allow for proper type casting of the `TotalCharges` column.
# MAGIC
# MAGIC > This step ensures that only relevant columns are included for feature engineering and model training.

# COMMAND ----------

from pyspark.sql.functions import when, col

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

telco_df = spark.read.option("nullValue", " ").csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

# Select columns of interest
telco_df = telco_df.select("gender", "SeniorCitizen", "Partner", "tenure", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod", "TotalCharges", "Churn")

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Splitting the Dataset into Training and Testing Sets
# MAGIC Once the data has been cleaned, we split it into training and testing sets using an 80-20 split.
# MAGIC > Since telco_df is a PySpark DataFrame, we will use `randomSplit()`.

# COMMAND ----------

train_df, test_df = telco_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transforming the Dataset
# MAGIC To ensure that all numerical and categorical features are compatible with machine learning algorithms, we perform several transformations.
# MAGIC
# MAGIC **Convert Integer and Boolean Columns to Double**
# MAGIC
# MAGIC - Many machine learning algorithms require numeric input, so we convert all **integer and boolean** columns to **double**.
# MAGIC     > This ensures numerical consistency in the dataset.

# COMMAND ----------

from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col, count, when


# Get a list of integer & boolean columns
integer_cols = [column.name for column in train_df.schema.fields if (column.dataType == IntegerType() or column.dataType == BooleanType())]

# Loop through integer columns to cast each one to double
for column in integer_cols:
    train_df = train_df.withColumn(column, col(column).cast("double"))
    test_df = test_df.withColumn(column, col(column).cast("double"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Identifying Missing Values**
# MAGIC
# MAGIC Handling missing data is crucial to prevent errors and bias in machine learning models. We first check for missing values in numerical and categorical columns.
# MAGIC
# MAGIC - **Find Numeric Columns with Missing Values**

# COMMAND ----------

from pyspark.sql.functions import count, when

# Identify numeric columns
num_cols = [c.name for c in train_df.schema.fields if c.dataType == DoubleType()]

# Count missing values in numeric columns
num_missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in num_cols]
row_dict_num = train_df.select(num_missing_values_logic).first().asDict()
num_missing_cols = [column for column in row_dict_num if row_dict_num[column] > 0]

print(f"Numeric columns with missing values: {num_missing_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC - **Find String Columns with Missing Values**

# COMMAND ----------

# Identify string columns
string_cols = [c.name for c in train_df.schema.fields if c.dataType == StringType()]

# Count missing values in string columns
string_missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in string_cols]
row_dict_string = train_df.select(string_missing_values_logic).first().asDict()
string_missing_cols = [column for column in row_dict_string if row_dict_string[column] > 0]

print(f"String columns with missing values: {string_missing_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a Feature Engineering Pipeline
# MAGIC
# MAGIC A **Spark ML Pipeline** chains multiple transformation and estimation steps into a single, reproducible workflow. Rather than applying each step manually - which risks inconsistency between training and test data - a pipeline ensures that the exact same transformations are learned from training data and then applied uniformly to new data.
# MAGIC
# MAGIC This is especially important for steps like **imputation** and **scaling**, where the statistics used for transformation (e.g., mean, standard deviation) must be derived from training data only and then applied to the test set. This prevents **data leakage**.
# MAGIC
# MAGIC **Our pipeline includes the following steps:**
# MAGIC
# MAGIC | Step | Transformer | Purpose |
# MAGIC |------|------------|---------|
# MAGIC | 1 | `StringIndexer` | Convert string categories to numeric indices |
# MAGIC | 2 | `Imputer` | Fill missing numerical values using the mean |
# MAGIC | 3 | `VectorAssembler` | Combine numerical columns into a single vector |
# MAGIC | 4 | `StandardScaler` | Normalize numerical feature values |
# MAGIC | 5 | `OneHotEncoder` | Convert categorical indices to binary sparse vectors |
# MAGIC | 6 | `VectorAssembler` | Combine all features into a final feature vector |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 - Encode Categorical Features: StringIndexer and OneHotEncoder
# MAGIC
# MAGIC Spark ML algorithms require numeric input. To handle string (categorical) columns, we use a two-step encoding approach:
# MAGIC
# MAGIC 1. **`StringIndexer`** assigns a stable numeric index for each known category. `handleInvalid="keep"` assigns null values or unseen categories a reserved index instead of failing. The `build_string_indexer_model()` helper (defined in the classroom setup) fits the indexer on a fixed label vocabulary — this keeps the serialized model small and within the Serverless v5 per-model size limit.
# MAGIC
# MAGIC 1. **`OneHotEncoder`** converts each numeric index into a sparse binary vector (e.g., index `1.0` in a 3-category column → `[0, 1, 0]`). This prevents the model from treating category indices as ordered or continuous numbers.
# MAGIC
# MAGIC > We define the categorical columns explicitly here, excluding the target column `Churn`, which we will handle separately.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

# Define categorical feature columns (excluding target 'Churn')
categorical_cols = ["gender", "Partner", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod"]

categorical_cols_indexed = [c + "_index" for c in categorical_cols]
ohe_cols = [c + "_ohe" for c in categorical_cols]

# Known label vocabulary for this Telco schema — passed to build_string_indexer_model()
TELCO_LABELS = {
    "gender":           ["Male", "Female"],
    "Partner":          ["No", "Yes"],
    "InternetService":  ["Fiber optic", "DSL", "No"],
    "Contract":         ["Month-to-month", "Two year", "One year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod":    ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)", "Credit card (automatic)"],
}

# Build a fixed-vocabulary StringIndexerModel (Serverless v5 compatible)
string_indexer_model = build_string_indexer_model(spark, categorical_cols, TELCO_LABELS)

# OneHotEncoder: convert numeric indices to sparse binary vectors
one_hot_encoder = OneHotEncoder(
    inputCols=categorical_cols_indexed,
    outputCols=ohe_cols
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 - Impute Missing Numerical Values
# MAGIC
# MAGIC Missing values in numerical columns can cause Spark ML estimators to fail. The `Imputer` transformer replaces missing values with a computed statistic - in this case, the **mean** of each column.
# MAGIC
# MAGIC > **Important:** The imputer is fitted on the **training set only**. When included in the pipeline, the mean values are computed from `train_df` during `pipeline.fit(train_df)` and then applied to both `train_df` and `test_df` during `.transform()`. This prevents data leakage from the test set into the training process.

# COMMAND ----------

from pyspark.ml.feature import Imputer

# Impute missing values in numeric columns
# outputCols matches inputCols - missing values are filled in-place
imputer = Imputer(
    inputCols=num_missing_cols,
    outputCols=num_missing_cols,
    strategy="mean"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 - Assemble and Scale Numerical Features
# MAGIC
# MAGIC Before scaling, we combine all numerical columns into a single vector using `VectorAssembler`. `StandardScaler` then standardizes the values by removing the mean and scaling to unit variance.
# MAGIC
# MAGIC > Scaling ensures that features with large numerical ranges (e.g., `TotalCharges`) do not dominate features with smaller ranges (e.g., `SeniorCitizen`) in distance-based or gradient-based algorithms.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler

# Assemble all numerical columns into a single vector
numerical_assembler = VectorAssembler(
    inputCols=num_cols,
    outputCol="numerical_assembled"
)

# Scale numerical features to standardize values
numerical_scaler = StandardScaler(
    inputCol="numerical_assembled",
    outputCol="numerical_scaled"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 - Assemble the Final Feature Vector
# MAGIC
# MAGIC The final `VectorAssembler` combines the scaled numerical features and the one-hot encoded categorical vectors into a **single feature vector** called `all_features`. Spark ML models expect all input features to be packed into one vector column in this format.

# COMMAND ----------

# Assemble scaled numerical + one-hot encoded categorical features into a single vector
feature_cols = ["numerical_scaled"] + ohe_cols

vector_assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="all_features"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 - Build the Pipeline
# MAGIC
# MAGIC We now combine all transformation steps into a single `Pipeline`. The order of stages matters - for example, `StringIndexer` must run before `OneHotEncoder`, and the numerical `VectorAssembler` must run before `StandardScaler`.
# MAGIC
# MAGIC Once defined, the pipeline can be **fitted once** on training data and **applied many times** to new data, making it reliable and production-ready.

# COMMAND ----------

from pyspark.ml import Pipeline

# Define the ordered sequence of pipeline stages.
# string_indexer_model is a single fitted StringIndexerModel that handles all 6
# categorical columns at once (inputCols / outputCols lists). Stage order matters:
#   StringIndexerModel → Imputer → num VectorAssembler → StandardScaler → OHE → final VectorAssembler
stages_list = [string_indexer_model, imputer, numerical_assembler, numerical_scaler, one_hot_encoder, vector_assembler]

# Instantiate the pipeline
pipeline = Pipeline(stages=stages_list)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Fit the Pipeline
# MAGIC
# MAGIC Fitting the pipeline on the training data runs each **Estimator** stage to learn its parameters (e.g., mean values for `Imputer`, scaling factors for `StandardScaler`). The `StringIndexerModel` was pre-built with a fixed vocabulary, so it acts as a Transformer and applies directly during fit. These learned or configured parameters are stored in the resulting `PipelineModel` and reused consistently when transforming new data.
# MAGIC
# MAGIC > **Transformers** (e.g., `VectorAssembler`, `OneHotEncoder`) do not learn parameters from data - they are applied directly.

# COMMAND ----------

# MAGIC %md
# MAGIC **What Happens During Fitting?**
# MAGIC
# MAGIC When we call `.fit(train_df)`, each stage in the pipeline processes the data in sequence:
# MAGIC
# MAGIC - **StringIndexerModel** maps each string category to a numeric index. The model was pre-built with a fixed Telco vocabulary using `build_string_indexer_model()`, so it acts as a Transformer during `pipeline.fit()` — no data scan is performed at this stage. This keeps the serialized `PipelineModel` within the Serverless v5 per-model size limit.
# MAGIC
# MAGIC - **Imputer** computes the **mean** for each specified numerical column from `train_df` and stores these values to fill missing data during transformation.
# MAGIC
# MAGIC - **VectorAssembler** (numerical) combines the specified numerical columns into a single vector — no learning required.
# MAGIC
# MAGIC - **StandardScaler** computes the **mean and standard deviation** of the assembled numerical vector from training data, used to normalize feature values.
# MAGIC
# MAGIC - **OneHotEncoder** converts numeric category indices into binary sparse vectors — no learning required.
# MAGIC
# MAGIC - **VectorAssembler** (final) merges all feature vectors into the single `all_features` column — no learning required.

# COMMAND ----------

# MAGIC %md
# MAGIC > **Note:** Spark Connect ML on Serverless v5 enforces a **256 MB per-model size limit**. The `build_string_indexer_model()` and `check_model_size()` helpers (defined in the classroom setup) keep this pipeline within that limit.

# COMMAND ----------

# Free any prior pipeline_model from the session cache before re-fitting (Serverless v5 — 1 GB session cache limit)
clear_ml_cache(globals(), ["pipeline_model"])

# Fit the pipeline on training data - learns all required statistics
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# Verify the fitted model stays within the Serverless v5 256 MB per-model size limit
check_model_size(pipeline_model, f"{DA.paths.working_dir}/_size_check_tmp")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Apply the Feature Engineering Pipeline
# MAGIC
# MAGIC Once the pipeline is **fitted** to the training data, it can be **applied to any dataset** using `.transform()`.
# MAGIC We apply the pipeline to both:
# MAGIC - **Train Dataset (`train_df`)** → Generates **transformed training features**.
# MAGIC - **Test Dataset (`test_df`)** → Ensures that the same transformations are applied consistently.
# MAGIC
# MAGIC The output is a **transformed dataset** containing the final `all_features` vector, ready for modeling.

# COMMAND ----------

# Transform both training and test datasets using the fitted pipeline
train_transformed_df = pipeline_model.transform(train_df)
test_transformed_df = pipeline_model.transform(test_df)

# COMMAND ----------

# Show a sample of the transformed feature vectors
train_transformed_df.select("all_features").show(3, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the Target Column
# MAGIC
# MAGIC Spark ML models require the **target (label) column to be numeric**. In our dataset, the `Churn` column contains string values - `"Yes"` or `"No"`. We convert these to a numeric representation before passing data to a model.
# MAGIC
# MAGIC We use the following mapping:
# MAGIC - `"Yes"` → `1.0` (customer churned)
# MAGIC - `"No"` → `0.0` (customer did not churn)
# MAGIC
# MAGIC > This step is applied **after** the pipeline transform because `Churn` was intentionally excluded from the feature engineering pipeline - it is our prediction target, not an input feature.

# COMMAND ----------

from pyspark.sql.functions import when, col

# Convert Churn string label to numeric (0.0 = No churn, 1.0 = Churned)
train_prepared_df = train_transformed_df.withColumn("label", when(col("Churn") == "Yes", 1.0).otherwise(0.0))
test_prepared_df = test_transformed_df.withColumn("label", when(col("Churn") == "Yes", 1.0).otherwise(0.0))

# Display the final prepared training dataset
display(train_prepared_df.select("all_features", "label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save and Reuse the Pipeline
# MAGIC
# MAGIC Preserving the feature engineering pipeline - including all learned parameters and transformation logic - is essential for maintaining reproducibility, enabling version control, and facilitating collaboration. In this section, we will:
# MAGIC
# MAGIC 1. **Save the Pipeline:** Save the fitted pipeline model to the designated working directory, organized within the **`spark_pipelines`** folder.
# MAGIC
# MAGIC 1. **Explore Loaded Pipeline Stages:** Upon loading the pipeline, inspect its stages to confirm the sequence of transformations that were applied.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Save the Pipeline

# COMMAND ----------

# Save the pipeline model with overwrite mode
pipeline_model.write().overwrite().save(f"{DA.paths.working_dir}/spark_pipelines")
print(f"Saved model to: {DA.paths.working_dir}/spark_pipelines")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Use Saved Model

# COMMAND ----------

# Load the saved pipeline model
from pyspark.ml import PipelineModel

loaded_pipeline = PipelineModel.load(f"{DA.paths.working_dir}/spark_pipelines")

# Show pipeline stages
loaded_pipeline.stages

# COMMAND ----------

# MAGIC %md
# MAGIC > **Using Saved Pipeline for Reuse**
# MAGIC >
# MAGIC > Although we already applied the pipeline earlier in this demo, we reload the saved pipeline and apply it again here to illustrate how saved pipelines can be reused in production.

# COMMAND ----------

# Use the loaded pipeline to transform the test dataset
test_transformed_df = loaded_pipeline.transform(test_df)
display(test_transformed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we built a structured feature engineering pipeline to streamline data preparation for machine learning. The pipeline handled missing value imputation using `Imputer`, encoded categorical features using fixed-vocabulary `StringIndexerModel` (equivalent role to `StringIndexer`) with `OneHotEncoder`, scaled numerical features with `StandardScaler`, and assembled all features into a single `all_features` vector ready for model training.
# MAGIC
# MAGIC We also prepared the target column by converting the `Churn` string label (`"Yes"` / `"No"`) to a numeric value (`1.0` / `0.0`) as required by Spark ML models.
# MAGIC
# MAGIC By fitting the pipeline on training data only and applying it consistently to both training and test sets, we ensured a reproducible and leakage-free feature transformation process. Saving the pipeline enables efficient reuse in future workflows and production deployments.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC