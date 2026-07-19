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
# MAGIC # Demo — Data Imputation and Transformation Pipeline
# MAGIC
# MAGIC In this demo, you will work through a complete data preparation workflow for machine learning.
# MAGIC
# MAGIC Starting from a raw, noisy dataset, you will progressively clean and transform the data—fixing data types, handling invalid values, addressing missing data, encoding categorical features, and preparing train and test datasets with proper scaling and imputation.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC - Coerce columns to the correct data types based on feature and target requirements  
# MAGIC - Identify and remove invalid or erroneous values in numeric columns  
# MAGIC - Analyze missing data and drop columns or rows based on defined thresholds  
# MAGIC - Impute boolean and string missing values using appropriate default strategies  
# MAGIC - Understand how Spark ML handles categorical missing values using `StringIndexer`  
# MAGIC - Encode categorical features using a `StringIndexer → OneHotEncoder → VectorAssembler` pipeline  
# MAGIC - Apply ordered indexing for ordinal categorical features  
# MAGIC - Split the dataset into training and test sets for modeling  
# MAGIC - Standardize numeric features using a scaler fit on the training set only  
# MAGIC - Impute numeric missing values using a strategy fit on the training set to avoid data leakage

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
# MAGIC <li><strong>Serverless Compute, Version 5</strong> — <a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976D2; text-decoration: underline;">How to select an environment version</a></li>
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

# MAGIC %run ../Includes/Classroom-Setup-2.1

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we will refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data Cleaning and Imputation
# MAGIC
# MAGIC We start by loading the raw dataset and walking through the full data preparation workflow: fixing data types, handling outliers, and addressing missing values. This prepares a clean dataset ready for feature engineering and modeling.

# COMMAND ----------

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-noisy' # CSV file name
dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

telco_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Coerce / Fix Data Types
# MAGIC
# MAGIC Even when Spark infers a schema, some columns may land in the wrong type. Correcting data types improves memory efficiency and ensures compatibility with Spark ML estimators.
# MAGIC
# MAGIC We will:
# MAGIC
# MAGIC * Convert **`SeniorCitizen`** and **`Churn`** binary columns to `BooleanType`.
# MAGIC * Convert **`tenure`** to `LongType` using `.selectExpr`.
# MAGIC * Convert **`Partner`**, **`Dependents`**, **`PhoneService`**, and **`PaperlessBilling`** to `BooleanType` using `spark.sql`.

# COMMAND ----------

from pyspark.sql.types import BooleanType, ShortType, IntegerType
from pyspark.sql.functions import col, when


binary_columns = ["SeniorCitizen", "Churn"]
telco_customer_churn_df = telco_df
for column in binary_columns:
    telco_customer_churn_df = telco_customer_churn_df.withColumn(column, col(column).cast(BooleanType()))

telco_customer_churn_df.select(*binary_columns).printSchema()

# COMMAND ----------

# PhoneService, PaperlessBilling, Partner, Dependents to boolean using spark.sql
telco_customer_churn_df.createOrReplaceTempView("telco_customer_churn_temp_view")

telco_customer_casted_df = spark.sql("""
    SELECT
        customerID,
        BOOLEAN(Dependents),
        BOOLEAN(Partner),
        BOOLEAN(PhoneService),
        BOOLEAN(PaperlessBilling),
        *
        EXCEPT (customerID, Dependents, Partner, PhoneService, PaperlessBilling, Churn),
        Churn
    FROM telco_customer_churn_temp_view
""")

telco_customer_casted_df.select("Dependents", "Partner", "PaperlessBilling", "PhoneService").printSchema()

# COMMAND ----------

# Tenure to Long/Integer using .selectExpr
telco_customer_casted_df = telco_customer_casted_df.selectExpr("* except(tenure)", "cast(tenure as long) tenure")
telco_customer_casted_df.select("tenure").printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Handle Invalid Values and Review Distributions
# MAGIC
# MAGIC Outliers can distort model training and statistical summaries. We address them in two steps:
# MAGIC
# MAGIC 1. **Remove rows with negative `TotalCharges`** — these are clearly erroneous values since charges cannot be negative.
# MAGIC 2. **Review the `PaymentMethod` distribution** — understanding how customers are spread across payment types helps inform encoding decisions later.

# COMMAND ----------

# MAGIC %md
# MAGIC **Filtering out negative TotalCharges**
# MAGIC
# MAGIC Negative values in `TotalCharges` are data quality issues. We remove these rows while preserving rows where `TotalCharges` is null (which will be handled separately in the imputation step).

# COMMAND ----------

from pyspark.sql.functions import col


# Remove customers with negative TotalCharges
TotalCharges_cutoff = 0

telco_no_outliers_df = telco_customer_casted_df.filter(
    (col("TotalCharges") > TotalCharges_cutoff) |
    (col("TotalCharges").isNull())  # Keep Nulls — handled in imputation step
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploring Payment Method Distribution**
# MAGIC
# MAGIC Let's review how customers are distributed across `PaymentMethod` categories. This is a useful step before encoding — it helps identify whether any categories are rare enough to require special handling.
# MAGIC
# MAGIC **Note:** In this dataset, all `PaymentMethod` categories are well-represented. We do not remove any rows based on frequency here. In a real-world scenario with a high-cardinality feature containing very rare categories, you might consider grouping them under a common label such as `"Other"` before encoding.

# COMMAND ----------

from pyspark.sql.functions import col, count, avg


group_var = "PaymentMethod"
stats_df = telco_no_outliers_df.groupBy(group_var) \
                      .agg(count("*").alias("Total"),
                           avg("MonthlyCharges").alias("MonthlyCharges")) \
                      .orderBy(col("Total").desc())

display(stats_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Save the cleaned dataset as the silver table before moving to missing value handling.

# COMMAND ----------

telco_customer_name_full = "telco_customer_full"
telco_customer_full_silver = f"{telco_customer_name_full}_silver"

telco_no_outliers_df.write.mode("overwrite").option("mergeSchema", True).saveAsTable(telco_customer_full_silver)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Handling Missing Values
# MAGIC
# MAGIC Missing data is one of the most common challenges in real-world ML datasets. Our approach:
# MAGIC
# MAGIC 1. **Identify** which columns have missing values and how many.
# MAGIC 2. **Drop columns** with more than 60% missing data — they carry too little signal to be useful.
# MAGIC 3. **Drop rows** where more than 80% of fields are null — these rows add little to model training.
# MAGIC 4. **Impute** remaining missing values with appropriate defaults.

# COMMAND ----------

from pyspark.sql.functions import (
    col,
    when,
    count,
    concat_ws,
    collect_list,
    sum,
    length,
    trim,
    lower,
)


def calculate_missing(input_df, show=True):
    """
    Helper function to calculate and display missing data per column.
    """
    missing_df_ = input_df.agg(
        *[
            sum(
                when(
                    col(c).isNull()
                    | (length(trim(col(c).cast("string"))) == 0)
                    | lower(trim(col(c).cast("string"))).isin(["none", "null"]),
                    1,
                ).otherwise(0)
            ).alias(c)
            for c in input_df.columns
        ]
    )

    def TransposeDF(df, columns, pivotCol):
        """Helper function to transpose spark dataframe"""
        columnsValue = list(
            map(lambda x: str("'") + str(x) + str("',") + str(x), columns)
        )
        stackCols = ",".join(x for x in columnsValue)
        df_1 = df.selectExpr(
            pivotCol, "stack(" + str(len(columns)) + "," + stackCols + ")"
        ).select(pivotCol, "col0", "col1")
        final_df = (
            df_1.groupBy(col("col0"))
            .pivot(pivotCol)
            .agg(concat_ws("", collect_list(col("col1"))))
            .withColumnRenamed("col0", pivotCol)
        )
        return final_df

    missing_df_out_T = TransposeDF(
        spark.createDataFrame([{"Column": "Number of Missing Values"}]).join(
            missing_df_
        ),
        missing_df_.columns,
        "Column",
    ).withColumn(
        "Number of Missing Values", col("Number of Missing Values").cast("long")
    )

    if show:
        display(missing_df_out_T.orderBy("Number of Missing Values", ascending=False))

    return missing_df_out_T


missing_df = calculate_missing(telco_no_outliers_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Drop columns with more than 60% missing data**
# MAGIC
# MAGIC Columns missing more than 60% of their values provide too little signal and introduce noise. We identify and drop them.

# COMMAND ----------

per_thresh = 0.6  # Drop if column has more than 60% missing data

N = telco_no_outliers_df.count()
to_drop_missing = [x.asDict()['Column'] for x in missing_df.select("Column").where(col("Number of Missing Values") / N >= per_thresh).collect()]

print(f"Dropping columns {to_drop_missing} for more than {per_thresh * 100}% missing data")
telco_no_missing_df = telco_no_outliers_df.drop(*to_drop_missing)
display(telco_no_missing_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Drop rows where most fields are missing**
# MAGIC
# MAGIC Rows with more than 80% of their fields missing provide very little information. We use `.na.drop()` with a threshold to remove them.

# COMMAND ----------

n_cols = len(telco_no_missing_df.columns)
telco_no_missing_df = telco_no_missing_df.na.drop(how='any', thresh=round(n_cols * .80))

print(f"Count — Before row drop: {telco_no_outliers_df.count()} / After: {telco_no_missing_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Impute Missing Data
# MAGIC
# MAGIC After dropping high-missing columns and sparse rows, we fill in the remaining missing values with sensible defaults.

# COMMAND ----------

# MAGIC %md
# MAGIC **Replace boolean missing values with `False`**

# COMMAND ----------

from pyspark.sql.types import BooleanType


bool_cols = [c.name for c in telco_no_missing_df.schema.fields if (c.dataType == BooleanType())]
telco_imputed_df = telco_no_missing_df.na.fill(value=False, subset=bool_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC **Replace string missing values with `No`**
# MAGIC
# MAGIC For service-related string columns (e.g., `OnlineSecurity`, `TechSupport`), a missing value typically means the service is not subscribed to. We impute these with `"No"`. Columns with meaningful categories (`gender`, `Contract`, `PaymentMethod`) are excluded and handled separately.

# COMMAND ----------

from pyspark.sql.types import StringType


to_exclude = ["customerID", "gender", "Contract", "PaymentMethod"]
string_cols = [c.name for c in telco_no_missing_df.drop(*to_exclude).schema.fields if c.dataType == StringType()]

telco_imputed_df = telco_imputed_df.na.fill(value='No', subset=string_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC **Handling missing values in categorical columns**
# MAGIC
# MAGIC For categorical columns such as `gender`, `Contract`, and `PaymentMethod`, we do not impute with a fixed string value because these columns have meaningful, distinct categories.
# MAGIC
# MAGIC Instead, Spark ML's `StringIndexer` handles nulls automatically when `handleInvalid` is set to `"keep"`. This treats null as its own separate category during indexing — an approach that works well for tree-based models, which can effectively learn from the presence of missing data as a signal.
# MAGIC
# MAGIC **When is mode imputation an alternative?**
# MAGIC Mode imputation (replacing nulls with the most frequent category) is appropriate when:
# MAGIC - The model cannot handle null-derived categories.
# MAGIC - Missing data is likely Missing At Random (MAR) and the mode is a reasonable proxy.
# MAGIC
# MAGIC In this demo, we rely on `StringIndexer`'s built-in null handling during the encoding step.

# COMMAND ----------

# Confirm remaining missing values after imputation
calculate_missing(telco_imputed_df)

# COMMAND ----------

telco_imputed_df.write.mode("overwrite").saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.telco_imputed_silver")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Encoding Categorical Features
# MAGIC
# MAGIC Most machine learning algorithms cannot accept raw string features. We need to convert categorical columns into numeric representations. Here we use the standard Spark MLlib workflow:
# MAGIC
# MAGIC **`StringIndexer → OneHotEncoder → VectorAssembler`**
# MAGIC
# MAGIC * **`StringIndexer`** — converts each unique string value to a numeric index.
# MAGIC * **`OneHotEncoder`** — converts the indexed column into a binary sparse vector, one position per category.
# MAGIC * **`VectorAssembler`** — combines one or more feature vectors into a single feature vector ready for a model.
# MAGIC
# MAGIC Setting `handleInvalid="keep"` in `StringIndexer` ensures that null values are treated as a separate category rather than causing an error.

# COMMAND ----------

sample_df = telco_imputed_df.select("Contract").distinct()
sample_df.show()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col


# StringIndexer
string_cols = ["Contract"]
index_cols = [column + "_index" for column in string_cols]

string_indexer = StringIndexer(inputCols=string_cols, outputCols=index_cols, handleInvalid="keep")
string_indexer_model = string_indexer.fit(sample_df)
indexed_df = string_indexer_model.transform(sample_df)

indexed_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Once indexed, we apply `OneHotEncoder` to produce binary vector representations.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder


ohe_cols = [column + "_ohe" for column in string_cols]

ohe = OneHotEncoder(inputCols=index_cols, outputCols=ohe_cols, handleInvalid="keep")
ohe_model = ohe.fit(indexed_df)
ohe_df = ohe_model.transform(indexed_df)
ohe_df.show()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler


selected_ohe_cols = ["Contract_ohe"]

assembler = VectorAssembler(inputCols=selected_ohe_cols, outputCol="features")
result_df_dense = assembler.transform(ohe_df)

result_df_dense.select("Contract", "features").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Ordered Indexing
# MAGIC
# MAGIC Some categorical columns are **ordinal** — they have a natural order that should be preserved. Standard one-hot encoding ignores this order. For example, `Contract` has a meaningful progression: `Month-to-month → One year → Two year`.
# MAGIC
# MAGIC For ordinal features used in tree-based models, manually mapping categories to ordered integer values can improve model interpretability and sometimes performance.

# COMMAND ----------

ordinal_cat = "Contract"
telco_imputed_df.select(ordinal_cat).distinct().show(truncate=False)

# COMMAND ----------

# Define the ordered category-to-index mapping
ordered_list = [
    "Month-to-month",
    "One year",
    "Two year"
]

ordinal_dict = {category: f"{index + 1}" for index, category in enumerate(ordered_list)}
print(ordinal_dict)

# COMMAND ----------

from pyspark.sql.functions import expr


ordinal_df = (
    telco_imputed_df
    .withColumn(f"{ordinal_cat}_ord", col(ordinal_cat))
    .replace(to_replace=ordinal_dict, subset=[f"{ordinal_cat}_ord"])
    .withColumn(f"{ordinal_cat}_ord", col(f"{ordinal_cat}_ord").cast('int'))
)

display(ordinal_df.select(ordinal_cat, f"{ordinal_cat}_ord"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Splitting Data
# MAGIC
# MAGIC Before applying any transformations that learn from the data (such as scaling or imputation), we split the dataset into training and test sets. This prevents **data leakage** — the test set must remain unseen during any fitting step.
# MAGIC
# MAGIC We use a reproducible 80/20 split with a fixed seed. Writing both sets as Delta tables ensures the split is stable across notebook runs.

# COMMAND ----------

# Split with 80% in train and 20% in test
train_df, test_df = telco_imputed_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# Materialize both sets as Delta tables
train_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.telco_customers_train")
test_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.telco_customers_baseline")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Standardize Features in a Training Set
# MAGIC
# MAGIC Feature scaling brings numeric features onto a comparable scale, which is important for distance-based and gradient-based models. We use `RobustScaler`, which is less sensitive to outliers than `StandardScaler` because it scales based on the interquartile range (IQR) rather than mean and standard deviation.
# MAGIC
# MAGIC **Key principle:** The scaler is **fit on the training set only**. The same fitted scaler is then applied to both training and test sets. This prevents test set statistics from influencing the transformation.

# COMMAND ----------

from pyspark.ml.feature import RobustScaler, VectorAssembler


num_cols_to_scale = ["MonthlyCharges"]
assembler = VectorAssembler().setInputCols(num_cols_to_scale).setOutputCol("numerical_assembled")

train_assembled_df = assembler.transform(train_df.select(*num_cols_to_scale))
test_assembled_df = assembler.transform(test_df.select(*num_cols_to_scale))

# Fit the scaler on the training set only
scaler = RobustScaler(inputCol="numerical_assembled", outputCol="numerical_scaled")
scaler_fitted = scaler.fit(train_assembled_df)

# Apply to both training and test sets
train_scaled_df = scaler_fitted.transform(train_assembled_df)
test_scaled_df = scaler_fitted.transform(test_assembled_df)

# COMMAND ----------

print("Training set (scaled):")
train_scaled_df.show(5)

# COMMAND ----------

print("Test set (scaled):")
test_scaled_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Numeric Missing Value Imputation
# MAGIC
# MAGIC Numeric features may still contain missing values after the earlier cleaning steps. Before passing data to a model, these must be filled with a representative value.
# MAGIC
# MAGIC **Mean vs. Median — when to use which:**
# MAGIC
# MAGIC | Strategy | When to Use |
# MAGIC |----------|-------------|
# MAGIC | **Mean** | Feature is approximately normally distributed with no significant outliers |
# MAGIC | **Median** | Feature is skewed or contains outliers — median is more robust |
# MAGIC
# MAGIC For `tenure`, which can be skewed toward newer or longer-tenured customers, we use the **median** strategy.
# MAGIC
# MAGIC As with scaling, the imputer must be **fit on the training set only** and then applied to both sets to avoid data leakage.

# COMMAND ----------

from pyspark.ml.feature import Imputer


# Define the imputer for tenure using median strategy
tenure_imputer = Imputer(
    inputCols=["tenure"],
    outputCols=["tenure_imputed"],
    strategy="median"
)

# Fit on training set only
tenure_imputer_fitted = tenure_imputer.fit(train_df.select("tenure"))

# Apply to both training and test sets
train_tenure_imputed_df = tenure_imputer_fitted.transform(train_df.select("tenure"))
test_tenure_imputed_df = tenure_imputer_fitted.transform(test_df.select("tenure"))

# COMMAND ----------

print("Training set — tenure imputed:")
train_tenure_imputed_df.show(5)

# COMMAND ----------

print("Test set — tenure imputed:")
test_tenure_imputed_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, you implemented an end-to-end data preparation workflow for machine learning using Spark.
# MAGIC
# MAGIC You were able to:
# MAGIC
# MAGIC - Correct data types to ensure compatibility with Spark ML  
# MAGIC - Identify and handle invalid values in numeric features  
# MAGIC - Analyze and address missing data using appropriate strategies  
# MAGIC - Encode categorical features using indexing and one-hot encoding  
# MAGIC - Split the dataset into training and test sets  
# MAGIC - Apply scaling and numeric imputation using transformations fit on the training data only  
# MAGIC
# MAGIC These steps form the foundation of a reliable and reproducible data preparation process. In the next demo, you will build on this work by assembling these transformations into a structured machine learning pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC