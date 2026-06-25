$ErrorActionPreference = "Stop"

echo "Running Graph Structural Analysis..."
.\.venv\Scripts\python.exe run_graph_structural_analysis.py

echo "Running ROC/AUC Analysis..."
.\.venv\Scripts\python.exe run_roc_analysis.py

echo "Running Feature Saliency..."
.\.venv\Scripts\python.exe run_feature_saliency.py

echo "Running UMAP Analysis..."
.\.venv\Scripts\python.exe run_umap_analysis.py

echo "Running SIGN Student Training..."
.\.venv\Scripts\python.exe phase5c_sign_student.py --sign-dir "artifacts/phase4_pyg" --model-dir "artifacts/phase5_model" --output-dir "artifacts/phase5_model" --epochs 30 --batch-size 4096 --kd-temp 3.0 --kd-alpha 0.7 --hidden-dim 512 --focal-gamma 2.0

echo "All complete!"
