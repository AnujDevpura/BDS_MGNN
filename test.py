from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower

INPUT_PATH = "data/*.csv"

spark = SparkSession.builder \
    .appName("LabelNormalizationDebug") \
    .getOrCreate()

df = spark.read.csv(
    INPUT_PATH,
    header=True,
    inferSchema=True
)

print("\n==============================")
print("RAW LABELS")
print("==============================")

df.groupBy(" Label").count().show(
    50,
    truncate=False
)

# ============================================
# NORMALIZE LABELS
# ============================================

df = df.withColumn(
    "clean_label",
    lower(trim(col(" Label")))
)

print("\n==============================")
print("NORMALIZED LABELS")
print("==============================")

df.groupBy("clean_label").count().show(
    50,
    truncate=False
)

# ============================================
# CREATE BINARY LABELS
# ============================================

df = df.withColumn(
    "label_binary",
    (col("clean_label") != "benign").cast("int")
)

print("\n==============================")
print("BINARY DISTRIBUTION")
print("==============================")

df.groupBy("label_binary").count().show()

spark.stop()