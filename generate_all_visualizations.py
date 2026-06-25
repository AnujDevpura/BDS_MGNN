import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_style():
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12

def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p, 'r') as f:
            return json.load(f)
    return None

def plot_training_history(history_file, title, out_path):
    data = load_json(history_file)
    if not data: return
    df = pd.DataFrame(data)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#ef4444'
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(df['epoch'], df['loss'], color=color, linewidth=2, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = '#10b981'
    ax2.set_ylabel('Validation Macro F1', color=color, fontweight='bold')
    ax2.plot(df['epoch'], df['val_macro_f1'], color=color, linewidth=2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(title, pad=20, fontweight='bold')
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_benchmark_comparison(bench_file, out_path):
    data = load_json(bench_file)
    if not data or 'results' not in data: return
    df = pd.DataFrame(data['results'])
    df = df.sort_values('macro_f1', ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = ['#10b981' if 'MGNN' in name else '#6366f1' for name in df['model']]
    bars = plt.barh(df['model'], df['macro_f1'], color=colors)
    
    plt.title("Model Benchmark Comparison (Macro F1)", pad=20, fontweight='bold')
    plt.xlabel("Test Macro F1 Score", fontweight='bold')
    plt.xlim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_latency_throughput(report_file, out_dir):
    data = load_json(report_file)
    if not data or 'results' not in data: return
    
    records = []
    for model_name, bs_dict in data['results'].items():
        if model_name == "sign_distilled":
            clean_name = "SIGN (Fast Graph)"
        elif model_name == "mgnn":
            clean_name = "MGNN (Graph AI)"
        else:
            clean_name = "MLP Baseline (Tabular)"
            
        for bs_key, metrics in bs_dict.items():
            if bs_key.startswith("bs_"):
                bs = int(bs_key.split("_")[1])
                records.append({
                    "Model": clean_name,
                    "Batch Size": bs,
                    "P95 Latency (ms)": metrics["end_to_end_latency"]["p95_ms"],
                    "Throughput": metrics["throughput_nodes_per_sec"]
                })
    
    df = pd.DataFrame(records)
    
    # Latency Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Batch Size", y="P95 Latency (ms)", hue="Model", 
                 marker="o", linewidth=2, palette=["#10b981", "#94a3b8", "#8b5cf6"])
    plt.xscale('log', base=2)
    plt.title("P95 Inference Latency vs Batch Size", pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_batch_size.png")
    plt.close()
    
    # Throughput Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Batch Size", y="Throughput", hue="Model", 
                 marker="o", linewidth=2, palette=["#10b981", "#94a3b8", "#8b5cf6"])
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.title("Inference Throughput vs Batch Size", pad=20, fontweight='bold')
    plt.ylabel("Throughput (Nodes / sec)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_vs_batch_size.png")
    plt.close()

def plot_feature_saliency(saliency_file, out_path):
    data = load_json(saliency_file)
    if not data or 'feature_importance' not in data: return
    
    top_features = data['feature_importance'][:20]
    df = pd.DataFrame(top_features)
    df = df.sort_values('importance_score', ascending=True)
    
    plt.figure(figsize=(12, 8))
    plt.barh(df['feature_name'], df['importance_score'], color='#3b82f6')
    plt.title("Top 20 Feature Saliency (Integrated Gradients)", pad=20, fontweight='bold')
    plt.xlabel("Relative Importance Score", fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(npy_file, title, out_path):
    p = Path(npy_file)
    if not p.exists(): return
    cm = np.load(p)
    cm_log = np.log1p(cm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_log, annot=cm, fmt='d', cmap='Blues', cbar=False)
    plt.title(title, pad=20, fontweight='bold')
    plt.ylabel("True Label", fontweight='bold')
    plt.xlabel("Predicted Label", fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_ablation_and_scaling(out_dir):
    # Component Ablation
    ablation_path = Path("artifacts/research/component_ablation.json")
    if ablation_path.exists():
        df = pd.DataFrame(load_json(ablation_path))
        df = df.sort_values('test_macro_f1', ascending=True)
        plt.figure(figsize=(10, 6))
        colors = ['#94a3b8' if v != 'Full MGNN v2 (baseline)' else '#10b981' for v in df['variant']]
        plt.barh(df['variant'], df['test_macro_f1'], color=colors)
        plt.axvline(x=0.8640, color='red', linestyle='--', alpha=0.5)
        plt.title("Component Ablation Study (Macro F1)", pad=20, fontweight='bold')
        plt.xlabel("Test Macro F1 Score", fontweight='bold')
        plt.xlim(0.70, 0.90)
        plt.tight_layout()
        plt.savefig(out_dir / "component_ablation.png")
        plt.close()

    # Graph Scaling
    scale_path = Path("artifacts/ablation/ablation_results.json")
    if scale_path.exists():
        df = pd.DataFrame(load_json(scale_path))
        plt.figure(figsize=(8, 5))
        plt.plot(df['nodes'], df['test_macro_f1'], marker='o', linewidth=3, markersize=10, color='#3b82f6')
        plt.title("Topological Scaling Phenomenon", pad=20, fontweight='bold')
        plt.xlabel("Number of Graph Nodes", fontweight='bold')
        plt.ylabel("Test Macro F1 Score", fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "graph_scaling.png")
        plt.close()

def main():
    setup_style()
    OUT_DIR = Path("artifacts/visualizations")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations to {OUT_DIR.absolute()} ...")
    
    # Training Convergence
    plot_training_history("artifacts/phase5_model/training_history.json", 
                          "MGNN v2 Training Convergence", OUT_DIR / "mgnn_training_convergence.png")
    plot_training_history("artifacts/phase5_model/sign_student_history.json", 
                          "SIGN Student Distillation Convergence", OUT_DIR / "sign_student_convergence.png")
    
    # Benchmarks & Inference
    plot_benchmark_comparison("artifacts/model_benchmarks/benchmark_results.json", OUT_DIR / "model_benchmark_f1.png")
    plot_latency_throughput("artifacts/benchmarks/realtime_inference_report.json", OUT_DIR)
    
    # Ablation & Scaling
    plot_ablation_and_scaling(OUT_DIR)
    
    # Confusion Matrices
    plot_confusion_matrix("artifacts/phase5_model/confusion_matrix.npy", 
                          "MGNN v2 Confusion Matrix (Log Scale)", OUT_DIR / "mgnn_confusion_matrix.png")
    plot_confusion_matrix("artifacts/phase5_model/sign_student_confusion_matrix.npy", 
                          "SIGN Student Confusion Matrix (Log Scale)", OUT_DIR / "sign_confusion_matrix.png")
                          
    # Feature Saliency
    plot_feature_saliency("artifacts/research/feature_saliency.json", OUT_DIR / "feature_saliency.png")
    
    print("\nSuccessfully generated all visualization plots!")

if __name__ == "__main__":
    main()
