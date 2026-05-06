# MGNN Big-Data Pipeline (Phase-wise)

#########################
cd "C:\My Projects\MGNN_BDS"
.\.venv\Scripts\Activate.ps1

$env:HADOOP_HOME="C:\hadoop"
${env:hadoop.home.dir}="C:\hadoop"
$env:Path="C:\hadoop\bin;$env:Path"

$env:PYSPARK_PYTHON="$PWD\.venv\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON="$PWD\.venv\Scripts\python.exe"

$env:PYSPARK_SUBMIT_ARGS="--master local[12] --driver-memory 12g --conf spark.executor.memory=12g --conf spark.sql.shuffle.partitions=64 --conf spark.default.parallelism=64 --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.2 --conf spark.hadoop.parquet.enable.dictionary=false pyspark-shell"
############################################################

python phase1_spark_prepare.py --input "data/*.csv" --output "artifacts/phase1_prepared_official" --replication-factor 3 --partitions 16


python phase2a_feature_engineering.py `
  --input "artifacts/phase1_output" `
  --output "artifacts/phase2_features" `
  --validation-output "artifacts/phase2_validation" `
  --scale
  
####################################################################################


This repository is organized as distributed Spark-first phases.

## UV environment setup (Windows PowerShell)

```powershell
# From repo root
uv venv .venv

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -r requirements.txt
```

## Files

- `phase1_spark_prepare.py`  
  Spark CSV ingest, cleaning, binary labels, scale simulation, repartitioning.
- `phase2_feature_engineering.py`  
  StringIndexer + VectorAssembler (+ optional scaling) to produce `features`.
- `phase3_graph_construction.py`  
  Distributed temporal edges + Spark LSH similarity edges.
- `phase4_export_for_pyg.py`  
  Export node features/labels/edges to PyTorch `.pt` tensors.
- `phase5_mgnn_loader.py`  
  Load exported tensors into a PyG `Data` object for MGNN training.

## Recommended run order

```powershell
# Phase 1
python phase1_spark_prepare.py `
  --input "data/*.csv" `
  --output "artifacts/phase1_prepared" `
  --label-col "Label" `
  --categorical-cols "" `
  --replication-factor 5 `
  --partitions 32

# Phase 2
python phase2_feature_engineering.py `
  --input "artifacts/phase1_prepared" `
  --output "artifacts/phase2_features" `
  --categorical-cols "" `
  --scale

# Phase 3
python phase3_graph_construction.py `
  --input "artifacts/phase2_features" `
  --output "artifacts/phase3_edges" `
  --temporal-window-seconds 10 `
  --lsh-distance-threshold 1.5 `
  --lsh-bucket-length 2.0 `
  --lsh-tables 3

# Phase 4 (optional max-nodes for memory-safe training)
python phase4_export_for_pyg.py `
  --feature-input "artifacts/phase2_features" `
  --edge-input "artifacts/phase3_edges" `
  --output-dir "artifacts/phase4_pyg" `
  --max-nodes 50000

# Phase 5
python phase5_mgnn_loader.py `
  --input-dir "artifacts/phase4_pyg" `
  --output "artifacts/phase5_data.pt"
```

## Notes

- This pipeline intentionally replaces sklearn k-NN graph building with Spark LSH.
- The existing MGNN model can now consume `x.pt`, `y.pt`, `edge_index.pt`, `edge_weight.pt`.
- Keep model architecture unchanged; only data ingestion/graph creation path is refactored.
- Your current CICIDS files do not include a usable timestamp column; `phase1_spark_prepare.py` auto-generates `event_time` when missing.
