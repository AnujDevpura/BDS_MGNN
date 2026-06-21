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
########################

1. Environment

cd "C:\My Projects\MGNN_BDS"

.\.venv\Scripts\Activate.ps1

$env:HADOOP_HOME="C:\hadoop"
${env:hadoop.home.dir}="C:\hadoop"
$env:Path="C:\hadoop\bin;$env:Path"

$env:PYSPARK_PYTHON="$PWD\.venv\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON="$PWD\.venv\Scripts\python.exe"


Install the new dependencies without using uv sync, because uv sync may remove the compiled PyG sampling extensions:
uv pip install -r requirements.txt

Verify the environment:

.\.venv\Scripts\python.exe -c "import torch, torch_geometric, pyg_lib, torch_sparse, torch_scatter, omegaconf; print(torch.__version__); print(torch.cuda.is_available()); print('Environment OK')"

If PyG extensions are missing:

uv pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv --find-links "https://data.pyg.org/whl/torch-2.5.0+cu121.html"


2. Full Data Pipeline


Phase 1: CICIDS preparation
Use replication factor 1 for the real dataset. Replication is only for synthetic scale experiments.

.\.venv\Scripts\python.exe phase1_spark_prepare.py `
  --input "data/*.csv" `
  --output "artifacts/phase1_output" `
  --replication-factor 1 `
  --partitions 64 `
  --master "local[12]" `
  --driver-memory "12g"


Phase 2A: Feature engineering
Scaling is recommended because the similarity graph uses feature geometry.

.\.venv\Scripts\python.exe phase2a_feature_engineering.py `
  --input "artifacts/phase1_output" `
  --output "artifacts/phase2_features" `
  --validation-output "artifacts/phase2_validation" `
  --scale `
  --master "local[12]" `
  --driver-memory "12g" `
  --partitions 64


Phase 2B: Balanced sampling

.\.venv\Scripts\python.exe phase2b_sample_nodes.py `
  --input "artifacts/phase2_features" `
  --output "artifacts/phase2_sampled_500k" `
  --master "local[12]" `
  --driver-memory "10g" `
  --shuffle-partitions 64


Phase 3A: Temporal graph

.\.venv\Scripts\python.exe phase3a_temporal.py `
  --input "artifacts/phase2_sampled_500k" `
  --output "artifacts/phase3_temporal" `
  --master "local[12]" `
  --driver-memory "12g" `
  --shuffle-partitions 96


Phase 3B: Distributed similarity graph
ANN searches in 50k-node batches.

.\.venv\Scripts\python.exe phase3b_similarity_fast.py `
  --input "artifacts/phase2_sampled_500k" `
  --output "artifacts/phase3_similarity" `
  --top-k 3 `
  --min-similarity 0.80 `
  --nlist 256 `
  --nprobe 16 `
  --search-batch-size 50000 `
  --faiss-threads 8 `
  --master "local[12]" `
  --driver-memory "12g" `
  --shuffle-partitions 96


Phase 3C: Heterogeneous graph fusion

.\.venv\Scripts\python.exe phase3c_combine.py `
  --temporal-input "artifacts/phase3_temporal" `
  --similarity-input "artifacts/phase3_similarity" `
  --output "artifacts/phase3_final" `
  --master "local[12]" `
  --driver-memory "10g" `
  --shuffle-partitions 96

Phase 4: PyG tensor export

.\.venv\Scripts\python.exe phase4_export_pyg.py `
  --feature-input "artifacts/phase2_sampled_500k" `
  --edge-input "artifacts/phase3_final" `
  --output-dir "artifacts/phase4_pyg" `
  --master "local[12]" `
  --driver-memory "12g" `
  --shuffle-partitions 96


3. Diagnostics
Graph structure:

.\.venv\Scripts\python.exe phase5c_graph_structural_diag.py `
  --input-dir "artifacts/phase4_pyg" `
  --output "artifacts/metrics/phase5c_graph_structural.json"

Tensor diagnostics:

.\.venv\Scripts\python.exe phase5b_diag.py

4. Main Training
This now:
Normalizes using training nodes only.
Computes class weights from training nodes only.
Saves train/validation/test indices.
Saves normalization statistics.
Supports checkpoint warm-starting.

.\.venv\Scripts\python.exe phase5_train_mgnn.py `
  --input-dir "artifacts/phase4_pyg" `
  --output-dir "artifacts/phase5_model" `
  --epochs 10 `
  --batch-size 1024 `
  --num-classes 15 `
  --metrics-output "artifacts/metrics/phase5_training.json"

Required new outputs:
artifacts/phase5_model/best_model.pt
artifacts/phase5_model/feature_normalization.pt
artifacts/phase5_model/split_indices.pt
artifacts/phase5_model/run_summary.json


5. Real-Time Benchmark

.\.venv\Scripts\python.exe realtime_inference_benchmark.py `
  --input-dir "artifacts/phase4_pyg" `
  --model-path "artifacts/phase5_model/best_model.pt" `
  --batch-sizes "1,8,32,128,512" `
  --warmup-steps 10 `
  --measure-steps 100 `
  --baseline-epochs 5 `
  --output "artifacts/benchmarks/realtime_inference_report.json"

Outputs:
artifacts/benchmarks/realtime_inference_report.json
artifacts/benchmarks/realtime_inference_report.csv
Use these comparison metrics:
End-to-end p95 latency
End-to-end p99 latency
Nodes/second throughput
Macro F1
Accuracy
Hardware and graph size
Compare other systems using the same hardware, held-out test indices, batch size, warmup count, and measurement count.


6. Streaming Windows

.\.venv\Scripts\python.exe run_streaming_windows.py `
  --input "artifacts/phase2_sampled_500k" `
  --output-root "artifacts/streaming" `
  --window-hours 6 `
  --step-hours 2 `
  --max-windows 12 `
  --master "local[12]"


7. Differential Training
Start with two windows as a smoke test:

.\.venv\Scripts\python.exe run_differential_training.py `
  --windows-root "artifacts/streaming" `
  --global-edge-input "artifacts/phase3_final" `
  --output-root "artifacts/diff_training" `
  --epochs-per-window 3 `
  --batch-size 1024 `
  --max-windows 2

Then run all windows:

.\.venv\Scripts\python.exe run_differential_training.py `
  --windows-root "artifacts/streaming" `
  --global-edge-input "artifacts/phase3_final" `
  --output-root "artifacts/diff_training" `
  --epochs-per-window 3 `
  --batch-size 1024

Report:
artifacts/diff_training/differential_training_report.json


8. Ablation-at-Scale
Run a smaller smoke test first:

.\.venv\Scripts\python.exe run_ablation_scale.py `
  --scales "100000" `
  --epochs 2 `
  --batch-size 1024

Full experiment:

.\.venv\Scripts\python.exe run_ablation_scale.py `
  --scales "100000,300000,550000" `
  --epochs 10 `
  --batch-size 1024

Final table:
artifacts/ablation/ablation_results.json
The current node universe contains approximately 549k nodes, so a 1M-node ablation requires increasing the Phase 2B sample size or using controlled replication.


BenchMarking 
Run all baselines:

.\.venv\Scripts\python.exe run_model_benchmarks.py `
  --models "xgboost,mlp,cnn,lstm,graphsage,gat" `
  --epochs 10 `
  --dense-batch-size 4096 `
  --graph-batch-size 1024 `
  --inference-batch-size 512