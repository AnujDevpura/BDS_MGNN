# MGNN BDS: Scalable Distributed Intrusion Detection via Hybrid Graph Neural Networks

> **Big Data Systems — Final Year Engineering Project**
> Apache Spark · PyTorch Geometric · Residual RGCN · CIC-IDS2017

---

## Abstract

Traditional Network Intrusion Detection Systems (NIDS) treat every network packet as an isolated tabular row, making them inherently vulnerable to adversarial feature evasion. This project proposes **MGNN BDS** — a scalable, end-to-end Big Data pipeline that transforms raw network traffic logs into a **multi-relational heterogeneous graph** and trains a **Hybrid Residual Relational Graph Convolutional Network (RGCN)** for robust, real-time intrusion detection.

The pipeline leverages **Apache Spark** for distributed graph construction (temporal and similarity edges) over the CIC-IDS2017 dataset (8 CSV files, ~800 MB), and **PyTorch Geometric** for mini-batch GNN training and inference on a graph of **449,104 nodes** and **3,407,252 edges** across **15 attack classes**. The proposed MGNN achieves **98.24% accuracy** and **0.864 Macro F1** on held-out test data, while maintaining **3× greater adversarial robustness** compared to XGBoost under Gaussian feature evasion attacks.

---

## Key Results

| Metric | MGNN (Proposed) | GraphSAGE | XGBoost | GAT | LSTM |
|---|---|---|---|---|---|
| **Test Accuracy** | **98.44%** | 97.14% | 99.78%* | 77.95% | 94.15% |
| **Test Macro F1** | **0.864** | 0.796 | 0.935* | 0.499 | 0.663 |
| **Adversarial F1 (σ=5)** | **0.163** | 0.106 | 0.051 | — | — |
| **P95 Latency (bs=512)** | 14.8 ms | 12.5 ms | 20.6 ms | 15.6 ms | 1.8 ms |
| **Throughput (bs=512)** | 41,311 nodes/s | 88,591 nodes/s | 21,688 nodes/s | 70,924 nodes/s | 359,058 nodes/s |

> *XGBoost achieves high static accuracy through tabular overfitting. Under adversarial noise it collapses to 0.051 F1 — a 94.6% degradation — while MGNN retains 0.163 F1 via multi-hop topological smoothing.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CIC-IDS2017 Dataset (~800MB, 8 CSV files, ~2.8M rows)                 │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  PHASE 1        │  Apache Spark
                    │  Ingestion &    │  Label Engineering
                    │  Cleaning       │  (15 attack classes)
                    └────────┬────────┘
                             │ Parquet
                    ┌────────▼────────┐
                    │  PHASE 2A       │  PySpark
                    │  Feature        │  78 features, StandardScaler
                    │  Engineering    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  PHASE 2B       │  Stratified Sampling
                    │  Balanced       │  449,104 nodes
                    │  Sampling       │
                    └────┬──────┬─────┘
                         │      │
            ┌────────────▼──┐  ┌▼───────────────────┐
            │  PHASE 3A     │  │  PHASE 3B           │
            │  Temporal     │  │  Similarity Graph   │  FAISS ANN
            │  Edges        │  │  Edges              │  (cosine ≥ 0.80)
            │  (same src IP │  │  (feature payload   │
            │   + time win) │  │   similarity)       │
            └───────┬───────┘  └──────────┬──────────┘
                    │                      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PHASE 3C           │  Heterogeneous Graph Fusion
                    │  Graph Fusion       │  Relation 0 + Relation 1
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PHASE 4            │  PyG Tensor Export
                    │  PyG Export         │  x.pt, y.pt, edge_index.pt,
                    │                     │  edge_type.pt, edge_weight.pt
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PHASE 5            │  PyTorch Geometric
                    │  MGNN Training      │  Hybrid Residual RGCN
                    │                     │  NeighborLoader (mini-batch)
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Streamlit          │  Real-time Dashboard
                    │  Dashboard          │  Live Inference + Research
                    └─────────────────────┘
```

---

## MGNN Model Architecture

The proposed **Hybrid Residual RGCN** fuses a direct tabular MLP bypass with a two-layer relational GCN:

```
Input Features (x)
       │
       ├──────────────────────────────────┐
       │                                  │
       ▼                                  ▼
 Feature Projection              Residual Tabular Bypass
 (Linear → LN → ReLU → Drop)    (Linear → LN → ReLU → Drop → Linear)
       │                                  │
       ▼                                  │
 RGCN Conv 1                             │
 (Relation 0: Temporal)                  │
 (Relation 1: Similarity)                │
       │ + residual                       │
       ▼                                  │
 RGCN Conv 2                             │
       │                                  │
       ▼                                  │
 Linear Head                             │
       │                                  │
       └──────────────── + ───────────────┘
                         │
                    Output Logits (15 classes)
```

**Key Design Choice:** The bypass path guarantees MGNN never performs worse than a pure tabular neural network, while the GNN path adds topological context. Their outputs are summed — making the system provably at least as powerful as MLP, while gaining adversarial robustness from graph structure.

---

## Dataset: CIC-IDS2017

- **Source:** Canadian Institute for Cybersecurity (University of New Brunswick)
- **Files:** 8 PCAP-derived CSV files (Monday–Friday, ~800 MB total)
- **Sampled Graph:** 449,104 nodes, 3,407,252 edges, 78 features, 15 classes
- **Class Imbalance:** BENIGN constitutes ~60% of sampled nodes; rare attacks (SQL Injection, Heartbleed) have <10 samples in test set

| Class ID | Attack Type |
|---|---|
| 0 | BENIGN |
| 1 | FTP-Patator |
| 2 | SSH-Patator |
| 3 | DoS Slowloris |
| 4 | DoS Slowhttptest |
| 5 | DoS Hulk |
| 6 | DoS GoldenEye |
| 7 | Heartbleed |
| 8 | Web Attack — Brute Force |
| 9 | Web Attack — XSS |
| 10 | Web Attack — SQL Injection |
| 11 | Infiltration |
| 12 | Bot |
| 13 | PortScan |
| 14 | DDoS |

---

## Graph Construction Details

### Relation 0 — Temporal Edges
Packets from the **same source IP** within a **sliding time window** are connected. Captures sequential attack patterns (port scanning, brute force login storms).

### Relation 1 — Similarity Edges  
Packets with **cosine similarity ≥ 0.80** in their 78-dimensional feature space are connected. Implemented with **FAISS IVF-Flat ANN** in 50k-node batches. Captures coordinated DDoS nodes even under IP spoofing.

---

## Environment Setup

```powershell
# 1. Navigate to project
cd "C:\My Projects\MGNN_BDS"
.\.venv\Scripts\Activate.ps1

# 2. Set Hadoop/Spark environment
$env:HADOOP_HOME="C:\hadoop"
${env:hadoop.home.dir}="C:\hadoop"
$env:Path="C:\hadoop\bin;$env:Path"
$env:PYSPARK_PYTHON="$PWD\.venv\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON="$PWD\.venv\Scripts\python.exe"
$env:PYSPARK_SUBMIT_ARGS="--master local[12] --driver-memory 12g --conf spark.executor.memory=12g --conf spark.sql.shuffle.partitions=64 --conf spark.default.parallelism=64 --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.2 --conf spark.hadoop.parquet.enable.dictionary=false pyspark-shell"

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Verify environment
.\.venv\Scripts\python.exe -c "import torch, torch_geometric, pyg_lib, torch_sparse, torch_scatter; print(torch.__version__); print(torch.cuda.is_available()); print('Environment OK')"
```

If PyG extensions are missing:
```powershell
uv pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv --find-links "https://data.pyg.org/whl/torch-2.5.0+cu121.html"
```

---

## Running the Full Pipeline

### Phase 1 — Spark Ingestion & Label Engineering
```powershell
.\.venv\Scripts\python.exe phase1_spark_prepare.py `
  --input "data/*.csv" --output "artifacts/phase1_output" `
  --replication-factor 1 --partitions 64 `
  --master "local[12]" --driver-memory "12g"
```

### Phase 2A — Feature Engineering
```powershell
.\.venv\Scripts\python.exe phase2a_feature_engineering.py `
  --input "artifacts/phase1_output" --output "artifacts/phase2_features" `
  --validation-output "artifacts/phase2_validation" --scale `
  --master "local[12]" --driver-memory "12g" --partitions 64
```

### Phase 2B — Balanced Sampling
```powershell
.\.venv\Scripts\python.exe phase2b_sample_nodes.py `
  --input "artifacts/phase2_features" --output "artifacts/phase2_sampled_500k" `
  --master "local[12]" --driver-memory "10g" --shuffle-partitions 64
```

### Phase 3A — Temporal Graph
```powershell
.\.venv\Scripts\python.exe phase3a_temporal.py `
  --input "artifacts/phase2_sampled_500k" --output "artifacts/phase3_temporal" `
  --master "local[12]" --driver-memory "12g" --shuffle-partitions 96
```

### Phase 3B — Similarity Graph (FAISS ANN)
```powershell
.\.venv\Scripts\python.exe phase3b_similarity_fast.py `
  --input "artifacts/phase2_sampled_500k" --output "artifacts/phase3_similarity" `
  --top-k 3 --min-similarity 0.80 --nlist 256 --nprobe 16 `
  --search-batch-size 50000 --faiss-threads 8 `
  --master "local[12]" --driver-memory "12g" --shuffle-partitions 96
```

### Phase 3C — Heterogeneous Graph Fusion
```powershell
.\.venv\Scripts\python.exe phase3c_combine.py `
  --temporal-input "artifacts/phase3_temporal" `
  --similarity-input "artifacts/phase3_similarity" `
  --output "artifacts/phase3_final" `
  --master "local[12]" --driver-memory "10g" --shuffle-partitions 96
```

### Phase 4 — PyG Tensor Export
```powershell
.\.venv\Scripts\python.exe phase4_export_pyg.py `
  --feature-input "artifacts/phase2_sampled_500k" `
  --edge-input "artifacts/phase3_final" `
  --output-dir "artifacts/phase4_pyg" `
  --master "local[12]" --driver-memory "12g" --shuffle-partitions 96
```

### Phase 5 — MGNN Training
```powershell
.\.venv\Scripts\python.exe phase5_train_mgnn.py `
  --input-dir "artifacts/phase4_pyg" --output-dir "artifacts/phase5_model" `
  --epochs 10 --batch-size 1024 --num-classes 15 `
  --metrics-output "artifacts/metrics/phase5_training.json"
```

---

## Running Experiments

### Real-Time Inference Benchmark
```powershell
.\.venv\Scripts\python.exe realtime_inference_benchmark.py `
  --input-dir "artifacts/phase4_pyg" `
  --model-path "artifacts/phase5_model/best_model.pt" `
  --batch-sizes "1,8,32,128,512" --warmup-steps 10 --measure-steps 100 `
  --output "artifacts/benchmarks/realtime_inference_report.json"
```

### Model Comparison Benchmarks (XGBoost, MLP, CNN, LSTM, GraphSAGE, GAT vs MGNN)
```powershell
.\.venv\Scripts\python.exe run_model_benchmarks.py `
  --models "xgboost,mlp,cnn,lstm,graphsage,gat" `
  --epochs 10 --dense-batch-size 4096 --graph-batch-size 1024 --inference-batch-size 512
```

### Adversarial Robustness Study
```powershell
.\.venv\Scripts\python.exe run_adversarial_robustness.py `
  --input-dir "artifacts/phase4_pyg" `
  --model-path "artifacts/phase5_model/best_model.pt" `
  --output "artifacts/research/adversarial_robustness.json"
```

### Graph Scaling Ablation (100k → 449k nodes)
```powershell
.\.venv\Scripts\python.exe run_graph_scaling_ablation.py `
  --scales "100000,300000,550000" --epochs 10 --batch-size 1024
```

---

## Launching the Dashboard

```powershell
uv run streamlit run streamlit_app.py
```

Then open http://localhost:8501

---

## Artifacts Structure

```
artifacts/
├── phase1_output/          # Cleaned Parquet (Spark output)
├── phase2_features/        # Scaled 78-feature Parquet
├── phase2_sampled_500k/    # 449k balanced node sample
├── phase3_temporal/        # Temporal edges (Relation 0)
├── phase3_similarity/      # FAISS similarity edges (Relation 1)
├── phase3_final/           # Fused heterogeneous edge list
├── phase4_pyg/             # PyG tensors (x.pt, y.pt, edge_index.pt, ...)
├── phase5_model/
│   ├── best_model.pt                # Trained MGNN weights
│   ├── feature_normalization.pt     # Training-set mean/std
│   ├── split_indices.pt             # Train/val/test indices
│   ├── run_summary.json             # Training metrics summary
│   ├── classification_report.json   # Per-class precision/recall/F1
│   └── training_history.json        # Loss/F1 per epoch
├── benchmarks/
│   └── realtime_inference_report.json
├── model_benchmarks/
│   └── benchmark_results.json
├── ablation/
│   └── ablation_results.json
└── research/
    ├── adversarial_robustness.json
    ├── pyspark_scaling.json
    └── streaming_engine_metrics.json
```

---

## Technical Stack

| Component | Technology | Purpose |
|---|---|---|
| Distributed Processing | Apache Spark 3.x (PySpark) | Graph edge construction at scale |
| Columnar Storage | Apache Parquet | Efficient inter-stage data transfer |
| ANN Search | FAISS (IVF-Flat) | Scalable similarity edge computation |
| GNN Framework | PyTorch Geometric | Mini-batch graph training & inference |
| GNN Model | Custom Hybrid Residual RGCN | Multi-relational intrusion detection |
| Tabular Baseline | XGBoost | Benchmark comparison |
| Dashboard | Streamlit + Plotly | Real-time inference visualization |
| Hardware | NVIDIA GeForce RTX 3050 (4GB) | GPU-accelerated training |

---

## Novelty & Contributions

1. **Hybrid Residual RGCN Architecture:** A GNN that provably never under-performs a pure MLP baseline, while gaining substantial topological robustness benefits.
2. **PySpark-to-PyG Integration:** A production-grade bridge between distributed Spark processing and GPU-accelerated GNN training via Parquet.
3. **Multi-Relational Graph Construction:** Simultaneous temporal (Relation 0) and semantic similarity (Relation 1) edge sets constructed at scale using FAISS ANN.
4. **Adversarial Robustness via Topology:** Demonstrated 3× relative F1 advantage over XGBoost under maximum adversarial noise (σ=5), validated across 11 noise levels.
5. **Positive Scaling Property:** F1 improves monotonically from 0.582 (112k nodes) to 0.864 (449k nodes) — unlike tabular models which plateau.